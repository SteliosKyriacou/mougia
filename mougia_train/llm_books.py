"""Krikri-8B trained on the BOOKS ONLY, in two stages (one LoRA, continued):

  stage 1 "books"   — continued pretraining (plain causal LM) on every Cypriot-dialect poem and distich from the
                      books (Sakellarios 1891, Cypriot Wikisource, anthology, song booklet…) → learn Cypriot, not SMG
  stage 2 "battles" — supervised call→reply on the battles reconstructed from those poems (speaker turns annotated by
                      hand): whole speeches and the couplet at each turn boundary → learn the question–reply structure

Data: data/old/processed/books_v1/ (books_prepare.py → annotation → books_build.py). Starts from the original
Krikri-8B-Instruct, not from the earlier fine-tune, so the result depends on the books alone.
Evaluation (logged per stage, same keys as the other runs): per-syllable val loss on held-out book poems (stage 1) and
on held-out book battles (stage 2), context gain, verse/raw reply quality and the share of replies that leak
Standard-Greek forms (smg_rate), on held-out book-battle calls AND on the usual v3 duel calls for comparison.

  .venv-llm/bin/python -I mougia_train/llm_books.py --run-id krikri-books
"""
import argparse
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
from mougia_train.llm_finetune import chat_ids, collate, llm_reply, load_examples, masked_nll  # noqa: E402
from mougia_train.runlog import RunLogger  # noqa: E402
from mougia_train.tokenizer import Tokenizer  # noqa: E402

BOOKS = os.path.join(ROOT, "data", "old", "processed", "books_v1")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Llama-Krikri-8B-Instruct-v1.5"))
    ap.add_argument("--books", default=BOOKS)
    ap.add_argument("--v3", default=os.path.join(ROOT, "data", "old", "processed", "train_v3"), help="for the v3 duel-call comparison")
    ap.add_argument("--run-id", default="krikri-books")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr-books", type=float, default=2e-4)
    ap.add_argument("--lr-battles", type=float, default=1e-4)
    ap.add_argument("--epochs-books", type=float, default=3.0)
    ap.add_argument("--epochs-battles", type=float, default=4.0)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--micro-batch-books", type=int, default=2, help="stage 1 scores every token, so smaller micro-batches")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--eval-every-min", type=float, default=30)
    ap.add_argument("--sample-every-min", type=float, default=15)
    ap.add_argument("--eval-replies", type=int, default=16)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--eval-every-steps", type=int, default=0, help="also evaluate every N optimizer steps (0 = time only)")
    ap.add_argument("--init-adapter", default=None, help="start from this LoRA adapter folder (e.g. a run's best-books)")
    ap.add_argument("--stages", default="books,battles", help="which stages to run, e.g. 'battles'")
    ap.add_argument("--log-every", type=int, default=5)
    return ap.parse_args()


def smg_leak(text):
    low = " " + re.sub(r"[^\w’']+", " ", text.lower()) + " "
    return any(g in low for g in SMG_GIVEAWAYS)


def _stop(*_):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    a = parse()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    read = lambda name: [json.loads(l) for l in open(os.path.join(a.books, name), encoding="utf-8")]
    books_train, books_val = read("stage1_train.jsonl"), read("stage1_val.jsonl")
    bat_train, bat_val = read("stage2_train.jsonl"), read("stage2_val.jsonl")
    fly_tok = Tokenizer.load(os.path.join(a.v3, "tokenizer.json"))
    _, _, v3_val_pairs, _, _ = load_examples(a.v3)
    v3_calls = [e["call"] for e in v3_val_pairs if e["kind"] == "duel"][: a.eval_replies]
    book_calls = [e["call"] for e in bat_val if e["level"] == "couplet"][: a.eval_replies]
    prompts = json.load(open(os.path.join(a.v3, "sample_prompts.json"), encoding="utf-8"))
    prompts += [{"id": f"book-{e['poem_id']}", "text": e["call"], "true_reply": e["reply"]} for e in bat_val if e["level"] == "couplet"][:2]

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
    if a.init_adapter:  # continue from an earlier run's adapter
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(os.path.join(a.init_adapter, "adapter_model.safetensors")))
    stats = json.load(open(os.path.join(a.books, "stats.json")))
    log = RunLogger(a.run_id, {
        "label": "Krikri-8B · books only (Cypriot LM → book battles)" if not a.init_adapter else
                 f"Krikri-8B · books only · {a.stages} continued from {os.path.basename(os.path.dirname(os.path.dirname(a.init_adapter)))}/{os.path.basename(a.init_adapter)}", "poet": "Krikri-8B", "kind": "llm_baseline",
        "args": vars(a), "model": os.path.basename(a.model), "data_dir": a.books, "data": stats,
        "params": {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "note": "losses are nats per syllable; smg_rate = share of replies with Standard-Greek forms",
    }, sample_every_s=a.sample_every_min * 60)

    syl = lambda text: max(1, sum(fly_tok.is_syllable(i) for i in fly_tok.encode(text)))

    def doc_items(doc):
        ids = tk(doc["text"] + tk.eos_token, add_special_tokens=False)["input_ids"]
        return [(ids[i:i + a.max_len], [1] * len(ids[i:i + a.max_len])) for i in range(0, len(ids), a.max_len) if len(ids[i:i + a.max_len]) > 8]

    book_chunks = [c for d in books_train for c in doc_items(d)]
    bat_items = [{"kind": "duel", "call": e["call"], "reply": e["reply"]} for e in bat_train]

    @torch.no_grad()
    def nll_per_syllable(items_and_text):
        model.eval()
        tot = n = 0.0
        for i in range(0, len(items_and_text), 4):
            chunk = items_and_text[i:i + 4]
            inp, mask, att = collate([x for x, _ in chunk], tk.pad_token_id)
            s, _ = masked_nll(model, inp.cuda(), mask.cuda(), att.cuda())
            tot += float(s.sum())
            n += sum(syl(t) for _, t in chunk)
        model.train()
        return tot / max(n, 1)

    def pair_loss(examples, calls=None):
        items = []
        for k, e in enumerate(examples):
            ex = {"kind": "duel", "call": calls[k] if calls else e["call"], "reply": e["reply"]}
            ids, m, _ = chat_ids(tk, ex, a.max_len)
            items.append(((ids, m), e["reply"]))
        return nll_per_syllable(items)

    def other_calls(ex):
        n = len(ex)
        out = []
        for i in range(n):
            j = (i + n // 2) % n
            while (ex[j]["poem_id"] == ex[i]["poem_id"] and n > 1 and j != (i - 1) % n) or j == i:
                j = (j + 1) % n
            out.append(ex[j]["call"])
        return out

    bat_val_c = [e for e in bat_val if e["level"] == "couplet"]

    def evaluate(stage):
        res = {"val_loss_books": nll_per_syllable([(x, d["text"]) for d in books_val for x in doc_items(d)])}
        if bat_val_c:
            res["val_loss_battles"] = pair_loss(bat_val_c)
            res["context_gain_battles"] = pair_loss(bat_val_c, other_calls(bat_val_c)) - res["val_loss_battles"]
        res["val_loss"] = res["val_loss_books"] if stage == "books" else res.get("val_loss_battles", res["val_loss_books"])
        model.eval()
        for tag, calls in (("", v3_calls), ("_books", book_calls)):
            for mode, suf in (("verse", ""), ("raw", "_raw")):
                outs = [llm_reply(model, tk, c, mode=mode, n=8) for c in calls]
                sc = [o["scores"] for o in outs]
                if not sc:
                    continue
                res[f"meter_rate{suf}{tag}"] = sum(s["meter_ok"] for s in sc) / len(sc)
                res[f"rhyme_rate{suf}{tag}"] = sum(s["rhyme_ok"] for s in sc) / len(sc)
                res[f"cy_rate{suf}{tag}"] = sum(s["cy_score"] >= 1.5 for s in sc) / len(sc)
                res[f"smg_rate{suf}{tag}"] = sum(smg_leak(o["reply"]) for o in outs) / len(outs)
        model.train()
        return res

    def sample(stage, step):
        model.eval()
        for p in prompts:
            v = llm_reply(model, tk, p["text"], mode="verse", n=16)
            log.sample(step, stage, prompt=p["text"], reply=v["reply"], prompt_id=p["id"], true_reply=p.get("true_reply"),
                       mode="verse (best of 16)", smg_leak=smg_leak(v["reply"]), **v["scores"])
        model.train()

    def save(tag):
        model.save_pretrained(os.path.join(log.dir, "checkpoints", tag))

    step, last_eval = 0, -1e18
    try:
        stages = [x.strip() for x in a.stages.split(",")]
        with log.sampling():
            log.eval(0, stages[0], **evaluate(stages[0]))
            sample(stages[0], 0)
        for stage, lr, epochs, pool in (("books", a.lr_books, a.epochs_books, "chunks"), ("battles", a.lr_battles, a.epochs_battles, "pairs")):
            if stage not in stages:
                continue
            items = book_chunks if pool == "chunks" else bat_items
            total = max(1, int(math.ceil(len(items) / (a.micro_batch * a.grad_accum)) * epochs))
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
            order, pos, best, bad, s0 = list(range(len(items))), 0, float("inf"), 0, step
            random.shuffle(order)
            t_last, tok_last = time.time(), 0
            while step - s0 < total:
                k = step - s0
                for g in opt.param_groups:
                    g["lr"] = lr * min(1.0, (k + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * k / total)))
                opt.zero_grad(set_to_none=True)
                loss_acc = 0.0
                mb = a.micro_batch_books if pool == "chunks" else a.micro_batch
                accum = a.grad_accum * a.micro_batch // mb  # same effective batch in both stages
                for _ in range(accum):
                    batch = []
                    for _ in range(mb):
                        if pos >= len(order):
                            random.shuffle(order)
                            pos = 0
                        it = items[order[pos]]
                        pos += 1
                        batch.append(it if pool == "chunks" else chat_ids(tk, it, a.max_len)[:2])
                    inp, mask, att = collate(batch, tk.pad_token_id)
                    s, n = masked_nll(model, inp.cuda(), mask.cuda(), att.cuda())
                    loss = s.sum() / n.sum().clamp_min(1)
                    (loss / accum).backward()
                    loss_acc += loss.item() / accum
                    tok_last += int(att.sum())
                gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                step += 1
                if step % a.log_every == 0:
                    torch.cuda.synchronize()
                    log.metrics(step, stage, train_loss=loss_acc, lr=opt.param_groups[0]["lr"], grad_norm=float(gn),
                                tokens_per_s=tok_last / max(time.time() - t_last, 1e-6),
                                gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9, phase_progress=(step - s0) / total)
                    t_last, tok_last = time.time(), 0
                if log.elapsed() - last_eval >= a.eval_every_min * 60 or (a.eval_every_steps and (step - s0) % a.eval_every_steps == 0):
                    with log.sampling():
                        res = evaluate(stage)
                        log.eval(step, stage, **res)
                    last_eval = log.elapsed()
                    if res["val_loss"] < best:
                        best, bad = res["val_loss"], 0
                        save(f"best-{stage}")
                    else:
                        bad += 1
                        if bad >= a.patience:
                            log.metrics(step, stage, event=f"early_stop after {bad} evals without improvement")
                            break
                if log.should_sample():
                    with log.sampling():
                        sample(stage, step)
            with log.sampling():
                res = evaluate(stage)
                log.eval(step, stage, **res)
                if res["val_loss"] < best:
                    save(f"best-{stage}")
                sample(stage, step)
            save(f"end-{stage}")
            if os.path.exists(os.path.join(log.dir, "checkpoints", f"best-{stage}")):  # continue from this stage's best
                from peft import set_peft_model_state_dict
                from safetensors.torch import load_file
                set_peft_model_state_dict(model, load_file(os.path.join(log.dir, "checkpoints", f"best-{stage}", "adapter_model.safetensors")))
            last_eval = -1e18
        log.close("finished")
    except KeyboardInterrupt:
        save("latest")
        log.close("stopped", note="interrupted")
    except Exception as e:
        log.close("failed", note=f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
