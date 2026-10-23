"""Fine-tune a pretrained Greek LLM (default: ILSP Llama-Krikri-8B-Instruct v1.5) on the same data as the fly brains.

Comparison baseline for the paper:
  - exactly the same training split as the fly runs (data/processed/train_v3: call→reply pairs, couplets,
    Cypriot dialect text), decoded from the fly tokenizer so nothing differs but the model;
  - chat format: the rival's couplet is the user turn, the reply couplet is the assistant turn (loss on the reply);
    couplets become "finish the couplet" (line 1 → line 2); dialect text is plain continued pretraining (30%);
  - 4-bit QLoRA (bf16 compute), LoRA on all attention + MLP projections;
  - losses reported per *syllable* (sum of LLM-token NLL over the reply / number of reply syllables), so they are
    comparable with the fly models' per-syllable losses; context gain measured the same way (true vs other call);
  - same verse-mode decoding (best of 16 ranked by rhyme > meter > dialect) and the same 4 sample prompts;
  - logs through RunLogger → appears in the browser's Training tab; best LoRA adapter kept by val duel loss.

Run with the LLM venv:
  .venv-llm/bin/python -I mougia_train/llm_finetune.py --run-id llm-krikri-8b
"""
import argparse
import math
import os
import random
import signal
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from mougia_train.generate import copy_ratio, rank, score_reply  # noqa: E402
from mougia_train.runlog import RunLogger  # noqa: E402
from mougia_train.tokenizer import CONT, DUEL, SEP, Tokenizer  # noqa: E402

SYSTEM = ("Είσαι ποιητάρης που την Κύπρο. Απαντάς σε τσιαττιστά με ένα δίστιχο: δύο στίχοι των δεκαπέντε συλλαβών "
          "που ομοιοκαταληκτούν, στην κυπριακή διάλεκτο, πάντα απαντώντας σε ό,τι σου είπαν.")
ASK = {"duel": "Απάντησε στο τσιαττιστό:\n{call}", "cont": "Συνέχισε το τραούδιν:\n{call}",
       "couplet": "Συμπλήρωσε το δίστιχο:\n{line}"}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Llama-Krikri-8B-Instruct-v1.5"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "processed", "train_v3"))
    ap.add_argument("--run-id", default="llm-krikri-8b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=2.0, help="passes over the pair + couplet examples")
    ap.add_argument("--lm-replay", type=float, default=0.3, help="share of micro-batches that are plain dialect text")
    ap.add_argument("--eval-every-min", type=float, default=30)
    ap.add_argument("--sample-every-min", type=float, default=15)
    ap.add_argument("--eval-replies", type=int, default=16)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=5)
    return ap.parse_args()


# ------------------------------------------------------------------------------------------------- data
def load_examples(data_dir):
    """Decode the fly training split back to text (same examples, same split)."""
    tok = Tokenizer.load(os.path.join(data_dir, "tokenizer.json"))
    dec = tok.decode

    def pairs(split):
        out = []
        for p in torch.load(os.path.join(data_dir, f"pairs_{split}.pt")):
            call = dec([t for t in p["call"] if t not in (SEP, DUEL, CONT)])
            out.append({"kind": p["style"], "call": call, "reply": dec(p["reply"]), "w": p.get("w", 1.0)})
        return out

    def couplets(split):
        out = []
        for ids in torch.load(os.path.join(data_dir, f"couplets_{split}.pt")):
            lines = dec(ids).split("\n")
            if len(lines) == 2:
                out.append({"kind": "couplet", "call": lines[0], "reply": lines[1], "w": 1.0})
        return out

    lm = torch.load(os.path.join(data_dir, "lm_train.pt")).long()
    return tok, pairs("train") + couplets("train"), pairs("val"), couplets("val"), lm


def chat_ids(tk, ex, max_len):
    """Token ids + loss mask for one chat example (loss on the assistant reply only)."""
    user = ASK["couplet"].format(line=ex["call"]) if ex["kind"] == "couplet" else ASK[ex["kind"]].format(call=ex["call"])
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    prompt = tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    p_ids = tk(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tk(ex["reply"] + tk.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + r_ids)[:max_len]
    mask = ([0] * len(p_ids) + [1] * len(r_ids))[:max_len]
    return ids, mask, len(p_ids)


def lm_ids(tk, fly_tok, lm, max_len):
    """A random window of Cypriot dialect text (decoded from the fly stream) for continued pretraining."""
    s = random.randint(0, len(lm) - 400)
    text = fly_tok.decode(lm[s:s + 300].tolist())
    ids = tk(text, add_special_tokens=False)["input_ids"][:max_len]
    return ids, [1] * len(ids)


def collate(items, pad_id):
    L = max(len(i) for i, _ in items)
    inp = torch.full((len(items), L), pad_id)
    mask = torch.zeros(len(items), L, dtype=torch.bool)
    att = torch.zeros(len(items), L, dtype=torch.long)
    for k, (ids, m) in enumerate(items):
        inp[k, :len(ids)] = torch.tensor(ids)
        mask[k, :len(m)] = torch.tensor(m, dtype=torch.bool)
        att[k, :len(ids)] = 1
    return inp, mask, att


def masked_nll(model, inp, mask, att):
    """Per-example summed NLL over masked target tokens (targets shifted by one). Only the masked positions are
    turned into float32 logits, so a 128k-word vocabulary doesn't blow up memory."""
    out = model(input_ids=inp, attention_mask=att)
    logits = out.logits[:, :-1]
    tgt = inp[:, 1:]
    m = mask[:, 1:]
    rows = torch.nonzero(m)[:, 0]
    sel_logits, sel_tgt = logits[m], tgt[m]
    # float32 cross-entropy in slices, so a long fully-scored text never needs all its fp32 logits at once
    nll = torch.cat([F.cross_entropy(sel_logits[i:i + 512].float(), sel_tgt[i:i + 512], reduction="none")
                     for i in range(0, len(sel_tgt), 512)]) if len(sel_tgt) else sel_logits.new_zeros(0, dtype=torch.float32)
    per_row = torch.zeros(inp.shape[0], device=inp.device, dtype=nll.dtype).index_add(0, rows, nll)
    return per_row, m.sum(1)


# ------------------------------------------------------------------------------------------------- generation
@torch.no_grad()
def llm_reply(model, tk, call, mode="verse", n=16, temperature=0.8, kind="duel"):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ASK[kind].format(call=call)}]
    prompt = tk.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    k = 1 if mode == "raw" else n
    out = model.generate(prompt.repeat(k, 1), attention_mask=torch.ones_like(prompt).repeat(k, 1), do_sample=True,
                         temperature=temperature, top_k=40, max_new_tokens=80, pad_token_id=tk.pad_token_id)
    seqs = out[:, prompt.shape[1]:]
    cands = []
    for i in range(0, k, 4):  # re-score the candidates in small batches (mean log-prob of the reply tokens)
        items = []
        for row in seqs[i:i + 4]:
            gen = [int(t) for t in row if int(t) != tk.pad_token_id]
            items.append((prompt[0].tolist() + gen, [0] * prompt.shape[1] + [1] * len(gen)))
        inp, mask, att = collate(items, tk.pad_token_id)
        nll, cnt = masked_nll(model, inp.to(model.device), mask.to(model.device), att.to(model.device))
        for (ids, m), s_, c_ in zip(items, nll.tolist(), cnt.tolist()):
            gen = ids[prompt.shape[1]:]
            text = tk.decode(gen, skip_special_tokens=True).strip()
            text = "\n".join(l.strip() for l in text.split("\n") if l.strip())[:300]
            cands.append({"reply": text, "logp": round(-s_ / max(c_, 1), 3), "scores": score_reply(text),
                          "copy": round(copy_ratio(text, call), 2)})
    cands.sort(key=rank, reverse=True)
    return {"reply": cands[0]["reply"], "scores": cands[0]["scores"], "logp": cands[0]["logp"], "candidates": cands}


# ------------------------------------------------------------------------------------------------- main
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

    fly_tok, train_ex, val_pairs, val_couplets, lm = load_examples(a.data)
    tk = AutoTokenizer.from_pretrained(a.model)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, device_map={"": 0}, torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    prompts = __import__("json").load(open(os.path.join(a.data, "sample_prompts.json"), encoding="utf-8"))

    log = RunLogger(a.run_id, {
        "label": f"Krikri-8B (pretrained Greek LLM, QLoRA)", "poet": "Krikri-8B", "kind": "llm_baseline", "args": vars(a),
        "model": os.path.basename(a.model), "params": {"total": total, "trainable": trainable},
        "data": {"train_examples": len(train_ex), "val_pairs": len(val_pairs), "val_couplets": len(val_couplets)},
        "data_dir": a.data, "note": "losses are nats per syllable (reply NLL summed over LLM tokens / reply syllables)",
    }, sample_every_s=a.sample_every_min * 60)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_ex) / (a.micro_batch * (1 - a.lm_replay)) / a.grad_accum)
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    def syllables_of(text):
        return max(1, sum(fly_tok.is_syllable(i) for i in fly_tok.encode(text)))

    @torch.no_grad()
    def eval_loss(examples, calls=None):
        """Per-syllable reply loss; calls overrides the call of each example (for the context-gain test)."""
        model.eval()
        tot = syl = 0.0
        for i in range(0, len(examples), 4):
            chunk = examples[i:i + 4]
            items = []
            for j, ex in enumerate(chunk):
                e = dict(ex)
                if calls is not None:
                    e["call"] = calls[i + j]
                ids, m, _ = chat_ids(tk, e, a.max_len)
                items.append((ids, m))
            inp, mask, att = collate(items, tk.pad_token_id)
            s, _ = masked_nll(model, inp.cuda(), mask.cuda(), att.cuda())
            tot += float(s.sum())
            syl += sum(syllables_of(ex["reply"]) for ex in chunk)
        model.train()
        return tot / syl

    def other_calls(ex):  # same rule as the fly runs: far away in the list, never this pair's own reply
        n = len(ex)
        out = []
        for i in range(n):
            j = (i + n // 2) % n
            while ex[j]["call"] == ex[i]["reply"] or j == i:
                j = (j + 1) % n
            out.append(ex[j]["call"])
        return out

    duel_val = [e for e in val_pairs if e["kind"] == "duel"]
    cont_val = [e for e in val_pairs if e["kind"] == "cont"][:64]
    coup_val = val_couplets[:64]

    def evaluate(step):
        res = {"val_loss_duel": eval_loss(duel_val), "val_loss_cont": eval_loss(cont_val), "val_loss_couplets": eval_loss(coup_val)}
        res["context_gain_duel"] = eval_loss(duel_val, other_calls(duel_val)) - res["val_loss_duel"]
        res["context_gain_cont"] = eval_loss(cont_val, other_calls(cont_val)) - res["val_loss_cont"]
        res["val_loss"] = res["val_loss_duel"]
        res["val_ppl"] = math.exp(min(res["val_loss"], 20))
        model.eval()
        calls = [e["call"] for e in duel_val[: a.eval_replies]]
        for mode, suffix in (("raw", "_raw"), ("verse", "")):
            sc = [llm_reply(model, tk, c, mode=mode, n=8)["scores"] for c in calls]
            res["meter_rate" + suffix] = sum(s["meter_ok"] for s in sc) / len(sc)
            res["rhyme_rate" + suffix] = sum(s["rhyme_ok"] for s in sc) / len(sc)
            ok = [s for s in sc if len(s["syllables"]) == 2]
            res["cy_rate" + suffix] = sum(s["cy_score"] >= 1.5 for s in ok) / len(sc)
            res["two_line_rate" + suffix] = len(ok) / len(sc)
        model.train()
        return res

    def sample(step):
        model.eval()
        for p in prompts:
            v = llm_reply(model, tk, p["text"], mode="verse", n=16)
            raw = llm_reply(model, tk, p["text"], mode="raw")
            log.sample(step, "finetune", prompt=p["text"], reply=v["reply"], prompt_id=p["id"], true_reply=p.get("true_reply"),
                       raw_reply=raw["reply"], mode="verse (best of 16)", **v["scores"])
        model.train()

    def save(tag):
        model.save_pretrained(os.path.join(log.dir, "checkpoints", tag))

    step, best, bad, last_eval = 0, float("inf"), 0, -1e18
    order = list(range(len(train_ex)))
    random.shuffle(order)
    pos = 0
    model.train()
    try:
        with log.sampling():
            log.eval(0, "finetune", **evaluate(0))
            sample(0)
        t_last, tok_last = time.time(), 0
        while step < total_steps:
            for g in opt.param_groups:
                g["lr"] = a.lr * sched(step)
            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for _ in range(a.grad_accum):
                if random.random() < a.lm_replay:
                    items = [lm_ids(tk, fly_tok, lm, a.max_len) for _ in range(a.micro_batch)]
                else:
                    items = []
                    for _ in range(a.micro_batch):
                        if pos >= len(order):
                            random.shuffle(order)
                            pos = 0
                        ids, m, _ = chat_ids(tk, train_ex[order[pos]], a.max_len)
                        items.append((ids, m))
                        pos += 1
                inp, mask, att = collate(items, tk.pad_token_id)
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
                dt = time.time() - t_last
                log.metrics(step, "finetune", train_loss=loss_acc, lr=opt.param_groups[0]["lr"], grad_norm=float(gn),
                            tokens_per_s=tok_last / max(dt, 1e-6), gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                            phase_progress=step / total_steps)
                t_last, tok_last = time.time(), 0
            if log.elapsed() - last_eval >= a.eval_every_min * 60:
                with log.sampling():
                    res = evaluate(step)
                    log.eval(step, "finetune", **res)
                last_eval = log.elapsed()
                if res["val_loss"] < best:
                    best, bad = res["val_loss"], 0
                    save("best")
                else:
                    bad += 1
                    if bad >= a.patience:
                        log.metrics(step, "finetune", event=f"early_stop after {bad} evals without improvement")
                        break
            if log.should_sample():
                with log.sampling():
                    sample(step)
        with log.sampling():
            res = evaluate(step)
            log.eval(step, "finetune", **res)
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
