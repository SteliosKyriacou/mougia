"""Krikri-8B continued pretraining on the books-only corpus (data/books/), for a fixed wall-clock budget.

Plain causal language modelling over the poems (verse after verse, poem after poem) — no prompts, no battles.
LoRA (4-bit QLoRA), learning rate warmed up then cosine-decayed over the whole time budget.
Every --eval-every-min: per-syllable loss on held-out poems (10%, split by poem), and generation checks —
  * couplet completion: the first line of a held-out couplet → the model writes the second line (meter, rhyme,
    Cypriot score, Standard-Greek leak);
  * duel replies to the usual v3 calls with the chat prompt (same metrics as the other Krikri runs).
Checkpoints: best (by held-out loss), a snapshot every --snapshot-every-hours, and end. Logged with RunLogger so it
shows in the web app's Training tab; adapters can be loaded in Chat.

  .venv-llm/bin/python -I mougia_train/llm_corpus.py --run-id krikri-books-24h --hours 24
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import signal
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from check_couplets import SMG_GIVEAWAYS  # noqa: E402
from mougia_train.generate import score_reply  # noqa: E402
from mougia_train.llm_finetune import collate, llm_reply, load_examples, masked_nll  # noqa: E402
from mougia_train.runlog import RunLogger  # noqa: E402
from mougia_train.tokenizer import Tokenizer  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Llama-Krikri-8B-Instruct-v1.5"))
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "books", "poems.jsonl"))
    ap.add_argument("--v3", default=os.path.join(ROOT, "data", "old", "processed", "train_v3"), help="duel calls + syllable tokenizer")
    ap.add_argument("--run-id", default="krikri-books-24h")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--eval-every-min", type=float, default=30)
    ap.add_argument("--sample-every-min", type=float, default=15)
    ap.add_argument("--snapshot-every-hours", type=float, default=2)
    ap.add_argument("--eval-replies", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=5)
    return ap.parse_args()


def smg_leak(text):
    low = " " + re.sub(r"[^\w’']+", " ", text.lower()) + " "
    return any(g in low for g in SMG_GIVEAWAYS)


def _stop(*_):
    raise KeyboardInterrupt


@torch.no_grad()
def complete_line(model, tk, first_line, temperature=0.8):
    """Raw continuation: given the first verse of a couplet, write the second (no chat template)."""
    ids = tk(first_line + "\n", add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=temperature, top_k=40,
                         max_new_tokens=48, pad_token_id=tk.pad_token_id)
    text = tk.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    second = next((l.strip() for l in text.split("\n") if l.strip()), "")
    return first_line + "\n" + second


def main():
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    a = parse()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    poems = [json.loads(l) for l in open(a.corpus, encoding="utf-8")]
    is_val = lambda p: int(hashlib.md5(p["poem_id"].encode()).hexdigest(), 16) % 1000 < a.val_frac * 1000
    train_p, val_p = [p for p in poems if not is_val(p)], [p for p in poems if is_val(p)]
    fly_tok = Tokenizer.load(os.path.join(a.v3, "tokenizer.json"))
    _, _, v3_val_pairs, _, _ = load_examples(a.v3)
    duel_calls = [e["call"] for e in v3_val_pairs if e["kind"] == "duel"][: a.eval_replies]
    first_lines = [p["lines"][i] for p in val_p for i in range(0, len(p["lines"]) - 1, 2)]
    random.Random(1).shuffle(first_lines)
    first_lines = first_lines[: a.eval_replies]
    prompts = json.load(open(os.path.join(a.v3, "sample_prompts.json"), encoding="utf-8"))

    tk = AutoTokenizer.from_pretrained(a.model)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, device_map={"": 0}, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))

    def chunks(p):
        ids = tk("\n".join(p["lines"]) + "\n\n", add_special_tokens=False)["input_ids"]
        return [ids[i:i + a.max_len] for i in range(0, len(ids), a.max_len) if len(ids[i:i + a.max_len]) > 8]

    train_chunks = [c for p in train_p for c in chunks(p)]
    val_items = [(c, "\n".join(p["lines"])) for p in val_p for c in chunks(p)]
    tokens_per_pass = sum(len(c) for c in train_chunks)
    log = RunLogger(a.run_id, {
        "label": f"Krikri-8B · books corpus ({a.hours:g} h)", "poet": "Krikri-8B", "kind": "llm_baseline", "args": vars(a),
        "model": os.path.basename(a.model), "data_dir": os.path.dirname(a.corpus),
        "data": {"train_poems": len(train_p), "val_poems": len(val_p), "train_tokens_per_pass": tokens_per_pass,
                 "train_lines": sum(len(p["lines"]) for p in train_p)},
        "params": {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "note": "plain continued pretraining on data/books; losses are nats per syllable; smg = Standard-Greek leak rate",
    }, sample_every_s=a.sample_every_min * 60)

    syl = lambda text: max(1, sum(fly_tok.is_syllable(i) for i in fly_tok.encode(text)))

    @torch.no_grad()
    def val_loss():
        model.eval()
        tot = n = 0.0
        for c, text in val_items:
            inp, mask, att = collate([(c, [1] * len(c))], tk.pad_token_id)
            s, _ = masked_nll(model, inp.cuda(), mask.cuda(), att.cuda())
            tot += float(s.sum())
        n = sum(syl(t) for t in {t for _, t in val_items})
        model.train()
        return tot / max(n, 1)

    def evaluate():
        res = {"val_loss_books": val_loss()}
        res["val_loss"] = res["val_loss_books"]
        res["val_ppl"] = math.exp(min(res["val_loss"], 20))
        model.eval()
        comp = [complete_line(model, tk, l) for l in first_lines]
        sc = [score_reply(c) for c in comp]
        res.update(meter_rate_completion=sum(s["meter_ok"] for s in sc) / len(sc),
                   rhyme_rate_completion=sum(s["rhyme_ok"] for s in sc) / len(sc),
                   cy_rate_completion=sum(s["cy_score"] >= 1.5 for s in sc) / len(sc),
                   smg_rate_completion=sum(smg_leak(c.split("\n", 1)[1]) for c in comp) / len(comp))
        for mode, suf in (("verse", ""), ("raw", "_raw")):
            outs = [llm_reply(model, tk, c, mode=mode, n=8) for c in duel_calls]
            s2 = [o["scores"] for o in outs]
            res[f"meter_rate{suf}"] = sum(s["meter_ok"] for s in s2) / len(s2)
            res[f"rhyme_rate{suf}"] = sum(s["rhyme_ok"] for s in s2) / len(s2)
            res[f"cy_rate{suf}"] = sum(s["cy_score"] >= 1.5 for s in s2) / len(s2)
            res[f"smg_rate{suf}"] = sum(smg_leak(o["reply"]) for o in outs) / len(outs)
        model.train()
        return res

    def sample(step):
        model.eval()
        for p in prompts:
            v = llm_reply(model, tk, p["text"], mode="verse", n=16)
            log.sample(step, "books", prompt=p["text"], reply=v["reply"], prompt_id=p["id"], mode="duel reply (verse, best of 16)",
                       smg_leak=smg_leak(v["reply"]), **v["scores"])
        for l in first_lines[:2]:
            c = complete_line(model, tk, l)
            log.sample(step, "books", prompt=l, reply=c, prompt_id="completion", mode="couplet completion (raw)",
                       smg_leak=smg_leak(c.split("\n", 1)[1]), **score_reply(c))
        model.train()

    def save(tag):
        model.save_pretrained(os.path.join(log.dir, "checkpoints", tag))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    budget = a.hours * 3600
    step, best, last_eval, next_snap = 0, float("inf"), -1e18, a.snapshot_every_hours * 3600
    order, pos = list(range(len(train_chunks))), 0
    random.shuffle(order)
    try:
        with log.sampling():
            log.eval(0, "books", **evaluate())
            sample(0)
        t_last, tok_last = time.time(), 0
        while log.elapsed() < budget:
            frac = min(1.0, log.elapsed() / budget)
            for g in opt.param_groups:
                g["lr"] = a.lr * min(1.0, (step + 1) / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for _ in range(a.grad_accum):
                batch = []
                for _ in range(a.micro_batch):
                    if pos >= len(order):
                        random.shuffle(order)
                        pos = 0
                    c = train_chunks[order[pos]]
                    pos += 1
                    batch.append((c, [1] * len(c)))
                inp, mask, att = collate(batch, tk.pad_token_id)
                s, n = masked_nll(model, inp.cuda(), mask.cuda(), att.cuda())
                loss = s.sum() / n.sum().clamp_min(1)
                (loss / a.grad_accum).backward()
                loss_acc += loss.item() / a.grad_accum
                tok_last += int(att.sum())
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1
            if step % a.log_every == 0:
                torch.cuda.synchronize()
                log.metrics(step, "books", train_loss=loss_acc, lr=opt.param_groups[0]["lr"], grad_norm=float(gn),
                            tokens_per_s=tok_last / max(time.time() - t_last, 1e-6), gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                            passes=step * a.grad_accum * a.micro_batch / len(train_chunks), phase_progress=frac)
                t_last, tok_last = time.time(), 0
            if log.elapsed() - last_eval >= a.eval_every_min * 60:
                with log.sampling():
                    res = evaluate()
                    log.eval(step, "books", **res)
                last_eval = log.elapsed()
                if res["val_loss"] < best:
                    best = res["val_loss"]
                    save("best")
            if log.should_sample():
                with log.sampling():
                    sample(step)
            if log.elapsed() >= next_snap:
                save(f"hour-{int(next_snap // 3600):02d}")
                next_snap += a.snapshot_every_hours * 3600
        with log.sampling():
            res = evaluate()
            log.eval(step, "books", **res)
            if res["val_loss"] < best:
                save("best")
            sample(step)
        save("end")
        log.close("finished")
    except KeyboardInterrupt:
        save("latest")
        log.close("stopped", note="interrupted")
    except Exception as e:
        log.close("failed", note=f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
