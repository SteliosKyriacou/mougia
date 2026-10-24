"""Fine-tune the pretrained diffusion LLM Dream-7B (Dream-org/Dream-v0-Instruct-7B) on the same data as the fly brains.

Diffusion counterpart of llm_finetune.py (Krikri-8B, autoregressive). Everything that can be the same is the same:
  - exactly the same examples (llm_finetune.load_examples on data/old/processed/train_v3: call→reply pairs, couplets as
    line 1 → line 2, 30% Cypriot dialect text replay), same SYSTEM / ASK prompts in Dream's (Qwen) chat template;
  - 4-bit QLoRA (nf4, bf16 compute), LoRA r=16 on all attention + MLP projections, gradient checkpointing;
  - same optimizer, schedule, early stopping and best adapter by val duel loss; same RunLogger logs, evaluation keys,
    verse-mode ranking (generate.rank) and the same 4 sample prompts.

What changes is the objective. Dream is a masked diffusion LM initialised from Qwen2.5-7B, and we follow its own SFT code
(HKUNLP/Dream src/trainer/fsdp_sft_trainer.py + src/diffllm/gen_utils.q_sample):
  - the reply is a fixed-length *canvas* of GEN_LEN tokens: reply + <|endoftext|>, right-padded with <|endoftext|>
    (Dream pads responses with EOS and keeps the padding in the loss, so the model learns where the reply ends; the
    canvas length is the same at generation time). Prompt tokens are never masked;
  - t ~ U(eps, 1) per example, every canvas token is replaced by <|mask|> with probability t;
  - logits are *shifted* as in Dream (the prediction for position i is read from the output at i-1, an artefact of the
    autoregressive initialisation), CE on masked positions, weighted by 1/t ("original" time reweighting),
    summed and divided by the number of canvas tokens;
  - dialect-text replay: a window of text is one canvas with no prompt (Dream's pre-training objective).

Losses for the paper are the diffusion ELBO of the reply canvas (an upper bound on the reply's NLL under the sampler's
canvas) in nats *per syllable*: E_t[ (1/t) Σ_masked -log p ] summed over the canvas, estimated with a fixed set of
stratified t values (ELBO_SAMPLES) and fixed per-example masking seeds, divided by the reply's syllable count from the
fly tokenizer (as llm_finetune). Context gain = ELBO with a mismatched call − ELBO with the true call (same masks).

Replies are sampled with Dream's own diffusion_generate (alg="entropy", GEN_STEPS steps over the GEN_LEN canvas).

Run with the LLM venv:
  .venv-llm/bin/python -I mougia_train/dream_finetune.py --run-id dream-7b
"""
import argparse
import json
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
from mougia_train.llm_finetune import ASK, SYSTEM, load_examples, lm_ids  # noqa: E402
from mougia_train.runlog import RunLogger  # noqa: E402

GEN_LEN = 104      # reply canvas (tokens): covers ~99.5% of train replies + EOS in Dream's tokenizer (Greek ≈ 1 tok/char)
GEN_STEPS = 32     # diffusion steps when sampling (≈3 tokens unmasked per step)
TIME_EPS = 1e-3    # t ~ U(eps, 1); the same eps Dream's sampler uses
ELBO_SAMPLES = 4   # t values per example for the validation ELBO


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Dream-v0-Instruct-7B"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "old", "processed", "train_v3"))
    ap.add_argument("--run-id", default="dream-7b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-prompt", type=int, default=384, help="prompt tokens kept (system + call, ≈290 in practice)")
    ap.add_argument("--gen-len", type=int, default=GEN_LEN, help="reply canvas length (train and sampling)")
    ap.add_argument("--lm-len", type=int, default=256, help="dialect-text canvas length (Krikri's max_len)")
    ap.add_argument("--epochs", type=float, default=2.0, help="passes over the pair + couplet examples")
    ap.add_argument("--lm-replay", type=float, default=0.3, help="share of micro-batches that are plain dialect text")
    ap.add_argument("--elbo-samples", type=int, default=ELBO_SAMPLES, help="t samples per example for the val ELBO")
    ap.add_argument("--eval-batch", type=int, default=8, help="rows per forward in the ELBO evaluation")
    ap.add_argument("--gen-steps", type=int, default=GEN_STEPS)
    ap.add_argument("--eval-n", type=int, default=8, help="verse-mode candidates in the eval (Krikri: 8)")
    ap.add_argument("--eval-every-min", type=float, default=60, help="Krikri: 30; a Dream eval costs ~25 min")
    ap.add_argument("--sample-every-min", type=float, default=15)
    ap.add_argument("--eval-replies", type=int, default=16)
    ap.add_argument("--patience", type=int, default=2, help="evals without improvement (Krikri: 3 x 30 min)")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = epochs decide; smoke tests)")
    ap.add_argument("--max-val", type=int, default=0, help="cap every val set at this size (0 = full; smoke tests only)")
    return ap.parse_args()


# ------------------------------------------------------------------------------------------------- data
def prompt_tokens(tk, kind, call, max_prompt=384):
    user = ASK[kind].format(call=call, line=call)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    prompt = tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tk(prompt, add_special_tokens=False)["input_ids"][-max_prompt:]


def canvas_tokens(tk, reply, gen_len):
    """reply + EOS, truncated / right-padded with EOS to gen_len (Dream SFT: padding is EOS and part of the target)."""
    ids = tk(reply + tk.eos_token, add_special_tokens=False)["input_ids"][:gen_len]
    return ids + [tk.eos_token_id] * (gen_len - len(ids))


def chat_item(tk, ex, gen_len, max_prompt=384):
    return prompt_tokens(tk, ex["kind"], ex["call"], max_prompt), canvas_tokens(tk, ex["reply"], gen_len)


def build_batch(items, pad_id, device):
    """items: [(prompt_ids, canvas_ids)] with equal canvas lengths. Prompts are left-padded (padding hidden from
    attention; position ids count real tokens only, as in Dream's sampler). Returns ids, attn (None if no padding),
    position ids, prompt width P (canvas = [:, P:])."""
    P = max(len(p) for p, _ in items)
    C = len(items[0][1])
    assert all(len(c) == C for _, c in items)
    B, L = len(items), P + C
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)
    for b, (p, c) in enumerate(items):
        ids[b, P - len(p):] = torch.tensor(p + c, dtype=torch.long)
        valid[b, :P - len(p)] = False
    pos = (valid.long().cumsum(-1) - 1).clamp_min(0)
    # key-padding mask only (padded query rows still attend to real tokens, so no all-masked rows → no NaNs)
    attn = None if bool(valid.all()) else valid[:, None, None, :].to(device)
    return ids.to(device), attn, pos.to(device), P


def _base(model):
    """The DreamModel inside a PeftModel (LoRA layers are injected in place)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def canvas_nll(model, ids, attn, pos, P, noisy, mask_id):
    """Per-row summed CE over the masked canvas positions. noisy: (B, C) bool, True = replaced by <|mask|>.
    Dream's shifted logits: the token at position j is predicted from the output at j-1 (position 0 from itself).
    Only the needed rows go through lm_head (152k vocab)."""
    m = _base(model)
    x = ids.clone()
    x[:, P:] = x[:, P:].masked_fill(noisy, mask_id)
    h = m.model(input_ids=x, attention_mask=attn, position_ids=pos, use_cache=False, return_dict=True).last_hidden_state
    rows, cols = torch.nonzero(noisy, as_tuple=True)
    if rows.numel() == 0:
        return torch.zeros(ids.shape[0], device=ids.device, dtype=torch.float32) + 0.0 * h.sum()
    logits = m.lm_head(h[rows, (P + cols - 1).clamp_min(0)].to(m.lm_head.weight.dtype)).float()
    ce = F.cross_entropy(logits, ids[rows, P + cols], reduction="none")
    return torch.zeros(ids.shape[0], device=ids.device, dtype=ce.dtype).index_add(0, rows, ce)


def diffusion_loss(model, items, tk, eps=TIME_EPS):
    """Dream SFT loss on one micro-batch: Σ_rows (1/t) Σ_masked CE / (number of canvas tokens)."""
    dev = next(p for p in model.parameters()).device
    ids, attn, pos, P = build_batch(items, tk.pad_token_id, dev)
    B, C = ids.shape[0], ids.shape[1] - P
    t = eps + (1 - eps) * torch.rand(B, device=dev)
    noisy = torch.rand(B, C, device=dev) < t[:, None]
    nll = canvas_nll(model, ids, attn, pos, P, noisy, tk.mask_token_id)
    return (nll / t).sum() / (B * C), int(ids.numel())


@torch.no_grad()
def elbo(model, tk, items, K, seeds, batch_rows=8):
    """Diffusion ELBO (nats, summed over the canvas) per item: mean over K stratified t's of (1/t) Σ_masked CE.
    Masks come from a generator seeded per item, so repeated evaluations (and true vs mismatched call) share them."""
    dev = next(p for p in model.parameters()).device
    ts = [(k + 0.5) / K for k in range(K)]
    rows = []  # (item index, t, noisy mask)
    for i, (item, sd) in enumerate(zip(items, seeds)):
        g = torch.Generator().manual_seed(int(sd))
        u = torch.rand(K, len(item[1]), generator=g)
        for k, t in enumerate(ts):
            rows.append((i, t, u[k] < t))
    out = torch.zeros(len(items), dtype=torch.float64)
    for s in range(0, len(rows), batch_rows):
        chunk = rows[s:s + batch_rows]
        ids, attn, pos, P = build_batch([items[i] for i, _, _ in chunk], tk.pad_token_id, dev)
        noisy = torch.stack([m for _, _, m in chunk]).to(dev)
        nll = canvas_nll(model, ids, attn, pos, P, noisy, tk.mask_token_id).double().cpu()
        for (i, t, _), v in zip(chunk, nll.tolist()):
            out[i] += v / t / K
    return out


# ------------------------------------------------------------------------------------------------- generation
def _clean(tk, row):
    ids = [int(t) for t in row]
    if tk.eos_token_id in ids:  # Dream's demo: the reply is everything before the first EOS
        ids = ids[:ids.index(tk.eos_token_id)]
    text = tk.decode(ids, skip_special_tokens=True).strip()
    return "\n".join(l.strip() for l in text.split("\n") if l.strip())[:300]


@torch.no_grad()
def dream_reply(model, tk, call, mode="verse", n=16, temperature=0.8, kind="duel", gen_len=GEN_LEN, steps=GEN_STEPS,
                score_samples=2):
    """Same contract as llm_finetune.llm_reply: dict(reply, scores, logp, candidates). Candidates are sampled with
    Dream's diffusion_generate (entropy-ordered unmasking); logp = −(ELBO of the sampled canvas) per reply token,
    from score_samples stratified t's (a cheap confidence score; only a tie-breaker in rank())."""
    prompt = prompt_tokens(tk, kind, call)
    dev = next(p for p in model.parameters()).device
    k = 1 if mode == "raw" else n
    x = torch.tensor([prompt], device=dev)
    seqs = []
    for i in range(0, k, 8):  # 8 candidates per diffusion run keeps the 152k-vocab logits small
        b = min(8, k - i)
        out = model.diffusion_generate(x.repeat(b, 1), max_new_tokens=gen_len, steps=steps, temperature=temperature,
                                       top_p=0.95, top_k=40, alg="entropy", alg_temp=0.0, mask_token_id=tk.mask_token_id,
                                       eos_token_id=tk.eos_token_id, pad_token_id=tk.pad_token_id,
                                       return_dict_in_generate=False, output_history=False)
        seqs += out[:, len(prompt):].tolist()
    items = [(prompt, s) for s in seqs]
    lp = elbo(model, tk, items, score_samples, list(range(len(items))), batch_rows=8) if score_samples else None
    cands = []
    for j, s in enumerate(seqs):
        text = _clean(tk, s)
        ntok = (s.index(tk.eos_token_id) if tk.eos_token_id in s else len(s)) + 1
        logp = -float(lp[j]) / ntok if lp is not None else 0.0
        cands.append({"reply": text, "logp": round(logp, 3), "scores": score_reply(text), "copy": round(copy_ratio(text, call), 2)})
    cands.sort(key=rank, reverse=True)
    return {"reply": cands[0]["reply"], "scores": cands[0]["scores"], "logp": cands[0]["logp"], "candidates": cands}


def load_dream(path, adapter=None, quantize=True, device=0):
    """Dream (4-bit nf4, bf16 compute) + tokenizer, optionally with a saved LoRA adapter (for the Chat tab)."""
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    tk = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16) if quantize else None
    model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device},
                                      quantization_config=q)
    model.config.use_cache = False  # no KV cache in diffusion; Dream's config.json says True
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tk


def other_calls(ex):  # same rule as llm_finetune / the fly runs: far away in the list, never this pair's own reply
    n = len(ex)
    out = []
    for i in range(n):
        j = (i + n // 2) % n
        while ex[j]["call"] == ex[i]["reply"] or j == i:
            j = (j + 1) % n
        out.append(ex[j]["call"])
    return out


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

    fly_tok, train_ex, val_pairs, val_couplets, lm = load_examples(a.data)
    model, tk = load_dream(a.model)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # prepare_model_for_kbit_training upcasts the frozen 152k-row embedding and lm_head to fp32 (+2.2 GB); they are
    # frozen, so keep them in bf16 to stay inside 12 GB (logits are still upcast to fp32 for the CE).
    for mod in (model.get_input_embeddings(), model.get_output_embeddings()):
        mod.to(torch.bfloat16)
    # task_type=None → plain PeftModel: Dream is not a transformers GenerationMixin causal LM
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    prompts = json.load(open(os.path.join(a.data, "sample_prompts.json"), encoding="utf-8"))

    log = RunLogger(a.run_id, {
        "label": "Dream-7B (pretrained diffusion LLM, QLoRA)", "poet": "Dream-7B", "kind": "dream_baseline", "args": vars(a),
        "model": os.path.basename(os.path.normpath(a.model)), "params": {"total": total, "trainable": trainable},
        "data": {"train_examples": len(train_ex), "val_pairs": len(val_pairs), "val_couplets": len(val_couplets)},
        "data_dir": a.data,
        "objective": "masked diffusion (Dream SFT): t~U(eps,1), mask reply canvas w.p. t, shifted-logit CE x 1/t",
        "note": (f"losses are nats per syllable: diffusion ELBO of the {a.gen_len}-token reply canvas (reply + EOS padding), "
                 f"{a.elbo_samples} stratified t samples with fixed masks, divided by reply syllables; "
                 f"replies sampled with diffusion_generate (entropy, {a.gen_steps} steps)"),
    }, sample_every_s=a.sample_every_min * 60)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_ex) / (a.micro_batch * (1 - a.lm_replay)) / a.grad_accum)
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    def syllables_of(text):
        return max(1, sum(fly_tok.is_syllable(i) for i in fly_tok.encode(text)))

    def eval_loss(examples, calls=None):
        """Per-syllable reply ELBO; calls overrides the call of each example (for the context-gain test)."""
        model.eval()
        items = []
        for i, ex in enumerate(examples):
            call = calls[i] if calls is not None else ex["call"]
            items.append(chat_item(tk, {**ex, "call": call}, a.gen_len, a.max_prompt))
        seeds = [1_000_003 * (a.seed + 1) + i for i in range(len(examples))]
        tot = float(elbo(model, tk, items, a.elbo_samples, seeds, a.eval_batch).sum())
        model.train()
        return tot / sum(syllables_of(ex["reply"]) for ex in examples)

    cap = (lambda xs: xs[:a.max_val]) if a.max_val else (lambda xs: xs)
    duel_val = cap([e for e in val_pairs if e["kind"] == "duel"])
    cont_val = cap([e for e in val_pairs if e["kind"] == "cont"][:64])
    coup_val = cap(val_couplets[:64])

    def gen(call, mode, n=16):
        return dream_reply(model, tk, call, mode=mode, n=n, gen_len=a.gen_len, steps=a.gen_steps)

    def evaluate(step):
        res = {"val_loss_duel": eval_loss(duel_val), "val_loss_cont": eval_loss(cont_val), "val_loss_couplets": eval_loss(coup_val)}
        res["context_gain_duel"] = eval_loss(duel_val, other_calls(duel_val)) - res["val_loss_duel"]
        res["context_gain_cont"] = eval_loss(cont_val, other_calls(cont_val)) - res["val_loss_cont"]
        res["val_loss"] = res["val_loss_duel"]
        res["val_ppl"] = math.exp(min(res["val_loss"], 20))
        model.eval()
        calls = [e["call"] for e in duel_val[: a.eval_replies]]
        for mode, suffix in (("raw", "_raw"), ("verse", "")):
            sc = [gen(c, mode, n=a.eval_n)["scores"] for c in calls]
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
            v = gen(p["text"], "verse", n=16)
            raw = gen(p["text"], "raw")
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
                    texts = [lm_ids(tk, fly_tok, lm, a.lm_len)[0] for _ in range(a.micro_batch)]
                    n_ = min(len(x) for x in texts)  # one canvas per window, no prompt, no padding
                    items = [([], x[:n_]) for x in texts]
                else:
                    items = []
                    for _ in range(a.micro_batch):
                        if pos >= len(order):
                            random.shuffle(order)
                            pos = 0
                        items.append(chat_item(tk, train_ex[order[pos]], a.gen_len, a.max_prompt))
                        pos += 1
                loss, ntok = diffusion_loss(model, items, tk)
                (loss / a.grad_accum).backward()
                loss_acc += loss.item() / a.grad_accum
                tok_last += ntok
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1
            if step % a.log_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.time() - t_last
                log.metrics(step, "finetune", train_loss=loss_acc, lr=opt.param_groups[0]["lr"], grad_norm=float(gn),
                            tokens_per_s=tok_last / max(dt, 1e-6),
                            gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
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
