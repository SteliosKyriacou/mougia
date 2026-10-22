"""MDLM baseline: a masked discrete diffusion language model trained from scratch on the same data as the fly brains.

Method (Sahoo et al. 2024, "Simple and Effective Masked Diffusion Language Models", arXiv 2406.07524):
  - absorbing-state diffusion: a token is replaced by [MASK] (id = vocab size) with probability t, linear schedule
    α_t = 1 − t, t ∈ [ε, 1], low-discrepancy (antithetic) t across the batch;
  - SUBS parameterisation: the denoiser never predicts [MASK] (it is not in the output vocabulary) and unmasked tokens
    are carried over unchanged, so only masked positions are scored;
  - loss = continuous-time NELBO: Σ_masked −log p_θ(x_i | z_t) · (1/t)  (α'_t/(1−α_t) = 1/t for the linear schedule),
    which upper-bounds the NLL of the noised block;
  - denoiser: bidirectional pre-LN transformer with RoPE, no time conditioning (unnecessary for absorbing-state MDLM);
  - sampling: ancestral steps t → s, each still-masked position is revealed with probability (t−s)/t from
    p_θ(x | z_t) (temperature + top-k like the fly runs); p_θ doesn't depend on t, so a forward pass is reused
    ("caching") whenever no token changed since the last one. All reply positions are decoded in parallel.

Sequences are [prefix | block]: the prefix (≤ --prefix-len tokens, left-padded, padding hidden from attention) is
never noised and carries no loss; the block (--reply-len tokens) is noised and must hold the whole answer, padded
with <pad> after <eos>, so the model learns where <nl> and <eos> go.
  pretrain : dialect-text windows — a random-length clean prefix (0–prefix-len tokens) + the next reply-len tokens
  couplets : [<bos>] + couplet, or (half the time) [<bos> line 1 <nl>] + line 2 <eos>
  pairs    : the call (<bos> … <sep><duel|cont>) + the reply (… <eos>); duel share and weights as in train.py
Losses (val_loss_*) are NELBO nats per *syllable*: Σ over the block's per-token ELBO terms (incl. the <nl>/<eos>/<pad>
positions) / number of syllable tokens of the target text, estimated with K stratified t per sequence (common random
numbers for the context-gain pair). They are upper bounds on the true NLL, so they are conservative against MDLM.

Run with the LLM venv:
  .venv-llm/bin/python -I mougia_train/mdlm_train.py --run-id mdlm-v3 --data data/processed/train_v3 --max-hours 4
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
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mougia_train.generate import copy_ratio, prompt_ids, rank, score_reply, verse_masks  # noqa: E402
from mougia_train.runlog import RunLogger  # noqa: E402
from mougia_train.tokenizer import BOS, CONT, DUEL, EOS, NL, PAD, SEP, UNK, Tokenizer  # noqa: E402

DATA = os.path.join(ROOT, "data", "processed", "train_v3")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="mdlm-v3")
    ap.add_argument("--data", default=None, help="training-data folder from prepare.py (default data/processed/train_v3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=640)
    ap.add_argument("--heads", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--prefix-len", type=int, default=64, help="clean prefix (the call) length, left-padded")
    ap.add_argument("--reply-len", type=int, default=48, help="noised block length (reply + <pad> after <eos>)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-3, help="smallest diffusion time")
    ap.add_argument("--lm-replay", type=float, default=0.3, help="share of dialect-text batches mixed into later phases")
    ap.add_argument("--duel-share", type=float, default=0.4, help="fraction of reply batches drawn from duel pairs")
    ap.add_argument("--couplet-prefix-share", type=float, default=0.5, help="couplets with line 1 as clean prefix")
    ap.add_argument("--patience", type=int, default=3, help="evals without val improvement before a phase stops early")
    ap.add_argument("--pretrain-tokens", type=float, default=300e6, help="noised block tokens in the dialect phase (≈50 passes; diffusion tolerates repetition, early stopping guards)")
    ap.add_argument("--couplet-tokens", type=float, default=30e6, help="block tokens in the couplet phase (early stopping may end sooner)")
    ap.add_argument("--pair-tokens", type=float, default=40e6, help="block tokens in the reply phase (early stopping may end sooner)")
    ap.add_argument("--elbo-k", type=int, default=8, help="t samples per sequence for validation NELBO")
    ap.add_argument("--steps", type=int, default=64, help="ancestral sampling steps")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--no-compile", action="store_true", help="don't torch.compile the training forward pass")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every-min", type=float, default=10)
    ap.add_argument("--sample-every-min", type=float, default=15)
    ap.add_argument("--ckpt-every-min", type=float, default=30)
    ap.add_argument("--eval-replies", type=int, default=16)
    ap.add_argument("--max-hours", type=float, default=0, help="stop after this much training time (0 = no limit)")
    return ap.parse_args()


# ---------------------------------------------------------------------------------------------------------------
# Denoiser
class Rotary(nn.Module):
    def __init__(self, dim, max_len=1024, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        f = torch.outer(torch.arange(max_len).float(), inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)

    def forward(self, x):  # x: B,H,T,D
        T = x.shape[2]
        c, s = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x, rope, attn_mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.mlp(self.ln2(x)))


class MDLM(nn.Module):
    def __init__(self, vocab, d=640, layers=12, heads=10, dropout=0.1):
        super().__init__()
        self.V = vocab
        self.mask_id = vocab  # [MASK] = vocab_size; it is an input token only (SUBS: never predicted)
        self.embed = nn.Embedding(vocab + 1, d)
        self.rope = Rotary(d // heads)
        self.blocks = nn.ModuleList(Block(d, heads, dropout) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.drop = nn.Dropout(dropout)
        self.apply(self._init)
        for b in self.blocks:  # GPT-2 style scaled init of residual projections
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * layers))
            nn.init.normal_(b.mlp[2].weight, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, x, keep, block):
        """x: B,T token ids (prefix + noised block); keep: B,T bool (False = left padding of the prefix, hidden from
        attention). Returns logits over the real vocabulary for the last `block` positions."""
        attn_mask = keep[:, None, None, :]
        h = self.drop(self.embed(x))
        for b in self.blocks:
            h = b(h, self.rope, attn_mask)
        return self.head(self.ln_f(h[:, -block:])).float()


# ---------------------------------------------------------------------------------------------------------------
# Data
def n_syl(tok_is_syl, ids):
    return max(1, sum(int(tok_is_syl[i]) for i in ids))


class Batches:
    def __init__(self, a, tok):
        self.a = a
        self.P, self.L = a.prefix_len, a.reply_len
        self.lm_train = torch.load(os.path.join(DATA, "lm_train.pt")).long()
        self.lm_val = torch.load(os.path.join(DATA, "lm_val.pt")).long()
        fits = lambda s: len(s) <= self.L
        # Training drops the few targets longer than the block (≈1% of couplets, 16 pairs); validation keeps all.
        self.c_train = [c for c in torch.load(os.path.join(DATA, "couplets_train.pt")) if fits(c[1:])]
        self.c_val = torch.load(os.path.join(DATA, "couplets_val.pt"))
        self.p_train = [p for p in torch.load(os.path.join(DATA, "pairs_train.pt")) if fits(p["reply"])]
        self.p_val = torch.load(os.path.join(DATA, "pairs_val.pt"))
        self.duel = [p for p in self.p_train if p["style"] == "duel"]
        self.cont = [p for p in self.p_train if p["style"] == "cont"]
        self.is_syl = [tok.is_syllable(i) for i in range(len(tok))]

    def pack(self, items, L=None):
        """items: (prefix ids, block ids). Returns x (B, P+L), keep mask, x0 block targets (B, L), syllables per item."""
        L = max([L or self.L] + [len(b) for _, b in items])
        B = len(items)
        x = torch.full((B, self.P + L), PAD, dtype=torch.long)
        keep = torch.zeros(B, self.P + L, dtype=torch.bool)
        keep[:, self.P:] = True
        syl = []
        for i, (pre, blk) in enumerate(items):
            pre = pre[-self.P:] if self.P else []
            if pre:
                x[i, self.P - len(pre):self.P] = torch.tensor(pre)
                keep[i, self.P - len(pre):self.P] = True
            x[i, self.P:self.P + len(blk)] = torch.tensor(blk)
            syl.append(n_syl(self.is_syl, blk))
        return x, keep, torch.tensor(syl, dtype=torch.float)

    def lm(self, split="train", n=None):
        B = n or self.a.batch
        if split == "train":
            items = []
            for s in torch.randint(0, len(self.lm_train) - self.P - self.L - 1, (B,)).tolist():
                k = random.randint(0, self.P)
                items.append((self.lm_train[s:s + k].tolist(), self.lm_train[s + k:s + k + self.L].tolist()))
            return self.pack(items)
        # validation: evenly spaced windows, a 16-token clean prefix + one block
        k = 16
        starts = torch.linspace(0, len(self.lm_val) - k - self.L - 1, B).long().tolist()
        return self.pack([(self.lm_val[s:s + k].tolist(), self.lm_val[s + k:s + k + self.L].tolist()) for s in starts])

    def couplet_item(self, c, line1_prefix):
        if line1_prefix and NL in c:
            j = c.index(NL) + 1
            return (c[:j], c[j:])
        return (c[:1], c[1:])  # [<bos>] + whole couplet

    def couplets(self, split="train", n=None):
        if split == "train":
            pick = random.sample(self.c_train, n or self.a.batch)
            return self.pack([self.couplet_item(c, random.random() < self.a.couplet_prefix_share) for c in pick])
        return self.pack([self.couplet_item(c, False) for c in self.c_val[: n or None]])

    def pairs(self, split="train", n=None, style=None, calls=None):
        if split == "train":
            B = n or self.a.batch
            k = sum(random.random() < self.a.duel_share for _ in range(B))
            pick = random.choices(self.duel, weights=[p["w"] for p in self.duel], k=k) + random.choices(self.cont, k=B - k)
            return self.pack([(p["call"], p["reply"]) for p in pick])
        pick = [p for p in self.p_val if style in (None, p["style"])][: n or None]
        calls = calls or [p["call"] for p in pick]
        return self.pack([(c, p["reply"]) for c, p in zip(calls, pick)])


# ---------------------------------------------------------------------------------------------------------------
# Diffusion
def noise(x0, t, mask_id, gen=None):
    """Absorbing forward process: each block token → [MASK] with probability t (linear schedule, α_t = 1 − t)."""
    u = torch.rand(x0.shape, device=x0.device, generator=gen)
    m = u < t[:, None]
    return torch.where(m, torch.full_like(x0, mask_id), x0), m


def nelbo_terms(model, x, keep, t, gen=None):
    """Per-position continuous-time NELBO terms of the block: CE on masked positions × 1/t. Returns B×L."""
    L = x.shape[1] - PREFIX_LEN
    x0 = x[:, -L:]
    zt, m = noise(x0, t, model.mask_id, gen)
    logits = model(torch.cat([x[:, :-L], zt], 1), keep, L)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x0.reshape(-1), reduction="none").view_as(x0)
    return ce * m / t[:, None]


PREFIX_LEN = 64  # set from --prefix-len in main()


def antithetic_t(B, eps, device):
    u = (torch.rand(1, device=device) + torch.arange(B, device=device) / B) % 1
    return eps + (1 - eps) * u


@torch.no_grad()
def eval_nelbo(model, batch, k, eps, dev, seed=1234, chunk=256):
    """Per-sequence NELBO (nats) averaged over k stratified t, and syllables per sequence. Fixed seed → common random
    numbers, so two calls with the same blocks (true vs mismatched call) see identical t and masks."""
    x, keep, syl = batch
    gen = torch.Generator(device=dev).manual_seed(seed)
    B = x.shape[0]
    tot = torch.zeros(B, device=dev)
    for j in range(k):
        t = eps + (1 - eps) * (j + torch.rand(B, device=dev, generator=gen)) / k
        for a in range(0, B, chunk):
            sl = slice(a, a + chunk)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tot[sl] += nelbo_terms(model, x[sl].to(dev), keep[sl].to(dev), t[sl], gen).sum(1)
    return tot / k, syl.to(dev)


@torch.no_grad()
def diffuse(model, prefix, n, L, steps, temperature=0.8, top_k=40, ban=(), dev="cuda"):
    """Ancestral sampling of n blocks for one prefix. Returns (block id lists, mean log-prob of the chosen tokens)."""
    P = PREFIX_LEN
    pre = prefix[-P:]
    x = torch.full((n, P + L), PAD, dtype=torch.long, device=dev)
    keep = torch.zeros(n, P + L, dtype=torch.bool, device=dev)
    keep[:, P:] = True
    if pre:
        x[:, P - len(pre):P] = torch.tensor(pre, device=dev)
        keep[:, P - len(pre):P] = True
    x[:, P:] = model.mask_id
    logp = torch.zeros(n, device=dev)
    probs, changed = None, True
    for i in range(steps):
        t, s = 1 - i / steps, 1 - (i + 1) / steps
        if changed or probs is None:  # caching: p_θ(x | z_t) only changes when z_t does
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, keep, L)
            logits = logits / max(temperature, 1e-5)
            if ban:
                logits[..., list(ban)] = -1e9
            if top_k:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits = logits.masked_fill(logits < v[..., -1:], -1e9)
            probs = F.softmax(logits.float(), -1)
        masked = x[:, P:] == model.mask_id
        reveal = masked & (torch.rand(n, L, device=dev) < (t - s) / t)
        changed = bool(reveal.any())
        if changed:
            draw = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(n, L)
            lp = torch.log(probs.gather(-1, draw[..., None])[..., 0].clamp_min(1e-12))
            logp += (lp * reveal).sum(1)
            x[:, P:] = torch.where(reveal, draw, x[:, P:])
    out = []
    for row in x[:, P:].tolist():
        if EOS in row:
            row = row[:row.index(EOS)]
        out.append([i for i in row if i != PAD])
    return out, (logp / L).tolist()


def mdlm_reply(model, tok, text, a, mode="verse", n=16, style="duel", masks=None, dev="cuda"):
    """Same contract as generate.reply: raw = 1 unconstrained sample; verse = best of n ranked by rank()."""
    ids = prompt_ids(tok, text, style)
    ban = (BOS, SEP, UNK, DUEL, CONT)  # <pad> stays allowed: it fills the block after <eos>
    if mode == "raw":
        seqs, lps = diffuse(model, ids, 1, a.reply_len, a.steps, a.temperature, a.top_k, ban, dev)
    else:
        masks = masks or verse_masks(tok)
        seqs, lps = diffuse(model, ids, n, a.reply_len, a.steps, a.temperature, a.top_k, ban + tuple(masks["ban"]), dev)
    cands = []
    for s, lp in zip(seqs, lps):
        t = tok.decode(s)
        cands.append({"reply": t, "logp": round(lp, 3), "scores": score_reply(t), "copy": round(copy_ratio(t, text), 2)})
    cands.sort(key=rank, reverse=True)
    return {"reply": cands[0]["reply"], "scores": cands[0]["scores"], "logp": cands[0]["logp"], "candidates": cands}


# ---------------------------------------------------------------------------------------------------------------
def _stop(*_):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    a = parse()
    global DATA, PREFIX_LEN
    if a.data:
        DATA = os.path.abspath(a.data)
    PREFIX_LEN = a.prefix_len
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"
    tok = Tokenizer.load(os.path.join(DATA, "tokenizer.json"))
    model = MDLM(len(tok), d=a.d_model, layers=a.layers, heads=a.heads, dropout=a.dropout).to(dev)
    fwd = model if a.no_compile else torch.compile(model)
    data = Batches(a, tok)
    masks = verse_masks(tok)
    prompts = json.load(open(os.path.join(DATA, "sample_prompts.json"), encoding="utf-8"))
    data_stats = json.load(open(os.path.join(DATA, "stats.json")))
    n_params = sum(p.numel() for p in model.parameters())

    log = RunLogger(a.run_id, {
        "label": "MDLM (masked diffusion, from scratch)", "poet": "MDLM", "kind": "mdlm_baseline", "args": vars(a),
        "model": {"type": "masked discrete diffusion LM (MDLM, Sahoo et al. 2024)", "layers": a.layers, "d_model": a.d_model,
                  "heads": a.heads, "positional": "RoPE", "attention": "bidirectional", "schedule": "linear (alpha_t = 1 - t)",
                  "loss": "continuous-time NELBO, SUBS parameterisation", "mask_id": len(tok),
                  "prefix_len": a.prefix_len, "reply_len": a.reply_len, "sampling_steps": a.steps},
        "params": {"total": n_params, "embedding": model.embed.weight.numel(), "head": model.head.weight.numel()},
        "data": data_stats, "data_dir": DATA, "vocab": len(tok), "sample_prompts": [p["id"] for p in prompts],
        "note": "val losses are NELBO upper bounds in nats per syllable (sum of the block's ELBO terms / target syllables)",
    }, sample_every_s=a.sample_every_min * 60)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "embed" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "embed" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98), fused=True)

    phases = [("pretrain", a.pretrain_tokens, data.lm), ("couplets", a.couplet_tokens, data.couplets), ("pairs", a.pair_tokens, data.pairs)]
    step, last_eval, last_ckpt = 0, -1e18, log.elapsed()

    def per_syl(batch):
        nelbo, syl = eval_nelbo(model, batch, a.elbo_k, a.eps, dev)
        return float(nelbo.sum() / syl.sum())

    def evaluate(phase):
        model.eval()
        out = {}
        out["val_loss_lm"] = per_syl(data.lm("val", n=256))
        out["val_loss_couplets"] = per_syl(data.couplets("val", n=256))
        out["val_loss_duel"] = per_syl(data.pairs("val", style="duel"))
        out["val_loss_cont"] = per_syl(data.pairs("val", style="cont", n=64))
        # Context gain: reply NELBO with someone else's call minus with the true call (nats/syllable), same rule as
        # train.py: the call from (i + n//2) % n, never the pair's own reply (in a duel the next call IS the reply).
        for style, n in (("duel", None), ("cont", 64)):
            ps = [p for p in data.p_val if p["style"] == style][:n]
            if len(ps) < 2:
                continue

            def other(i):
                j = (i + len(ps) // 2) % len(ps)
                while ps[j]["call"][1:-2] == ps[i]["reply"][:-1] or j == i:
                    j = (j + 1) % len(ps)
                return ps[j]["call"]
            true = per_syl(data.pairs("val", style=style, n=n))
            wrong = per_syl(data.pairs("val", style=style, n=n, calls=[other(i) for i in range(len(ps))]))
            out[f"context_gain_{style}"] = wrong - true
        key = {"pretrain": "val_loss_lm", "couplets": "val_loss_couplets", "pairs": "val_loss_duel"}[phase]
        out["val_loss"] = out[key]
        out["val_ppl"] = math.exp(min(out["val_loss"], 20))
        calls = [tok.decode(p["call"]) for p in data.p_val if p["style"] == "duel"][: a.eval_replies]
        for mode, suffix in (("raw", "_raw"), ("verse", "")):
            scores = [mdlm_reply(model, tok, c, a, mode=mode, n=16, masks=masks)["scores"] for c in calls]
            out["meter_rate" + suffix] = sum(s["meter_ok"] for s in scores) / len(scores)
            out["rhyme_rate" + suffix] = sum(s["rhyme_ok"] for s in scores) / len(scores)
            ok = [s for s in scores if len(s["syllables"]) == 2]
            out["cy_rate" + suffix] = sum(s["cy_score"] >= 1.5 for s in ok) / len(scores)
            out["two_line_rate" + suffix] = len(ok) / len(scores)
        model.train()
        return out

    def sample(phase):
        model.eval()
        for p in prompts:
            v = mdlm_reply(model, tok, p["text"], a, mode="verse", n=16, masks=masks)
            raw = mdlm_reply(model, tok, p["text"], a, mode="raw")
            log.sample(step, phase, prompt=p["text"], reply=v["reply"], prompt_id=p["id"], true_reply=p.get("true_reply"),
                       raw_reply=raw["reply"], mode="verse (best of 16)", **v["scores"])
        model.train()

    def save(tag):
        torch.save({"model": model.state_dict(), "step": step, "args": vars(a), "vocab": len(tok)},
                   os.path.join(log.dir, "checkpoints", f"{tag}.pt"))

    model.train()
    phase = phases[0][0]
    try:
        best = {}

        def eval_and_track(phase):
            res = evaluate(phase)
            log.eval(step, phase, **res)
            if res["val_loss"] < best.get(phase, float("inf")):
                best[phase] = res["val_loss"]
                save(f"best-{phase}")
                return True
            return False

        for phase, budget, batch_fn in phases:
            seen, phase_start, t_last, tok_last, bad = 0, step, time.time(), 0, 0
            block_tokens = a.batch * a.reply_len
            replay_share = a.lm_replay if phase != "pretrain" else 0
            total_steps = max(1, int(budget / block_tokens / (1 - replay_share)))
            while seen < budget:
                replay = phase != "pretrain" and random.random() < a.lm_replay
                x, keep, _ = data.lm("train") if replay else batch_fn("train")
                x, keep = x.to(dev, non_blocking=True), keep.to(dev, non_blocking=True)
                k = step - phase_start
                scale = min(1.0, (k + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, k / total_steps))))
                for g in opt.param_groups:
                    g["lr"] = a.lr * scale
                t = antithetic_t(x.shape[0], a.eps, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = nelbo_terms(fwd, x, keep, t).mean()  # nats per block token
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                if not torch.isfinite(gn):
                    log.metrics(step, phase, event="nonfinite_grad")
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                step += 1
                n_tok = 0 if replay else block_tokens
                seen += n_tok
                tok_last += block_tokens
                if step % a.log_every == 0:
                    torch.cuda.synchronize()
                    dt = time.time() - t_last
                    log.metrics(step, phase, train_loss=loss.item(), lr=opt.param_groups[0]["lr"], grad_norm=float(gn), replay=replay,
                                tokens_per_s=tok_last / max(dt, 1e-6), gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                                phase_progress=seen / budget)
                    t_last, tok_last = time.time(), 0
                if log.elapsed() - last_eval >= a.eval_every_min * 60:
                    with log.sampling():
                        improved = eval_and_track(phase)
                    last_eval = log.elapsed()
                    bad = 0 if improved else bad + 1
                    if bad >= a.patience:  # unlike train.py also in pretraining: ≈50 passes can overfit
                        log.metrics(step, phase, event=f"early_stop after {bad} evals without improvement")
                        break
                if log.should_sample():
                    with log.sampling():
                        sample(phase)
                if log.elapsed() - last_ckpt >= a.ckpt_every_min * 60:
                    save("latest")
                    last_ckpt = log.elapsed()
                if a.max_hours and log.elapsed() > a.max_hours * 3600:
                    raise TimeoutError
            with log.sampling():
                eval_and_track(phase)
                sample(phase)
            save(f"end-{phase}")
            if phase in best:  # continue the next phase from this phase's best weights
                model.load_state_dict(torch.load(os.path.join(log.dir, "checkpoints", f"best-{phase}.pt"), weights_only=False)["model"])
        log.close("finished")
    except TimeoutError:
        with log.sampling():
            log.eval(step, phase, **evaluate(phase))
            sample(phase)
        save("latest")
        log.close("stopped", note=f"max-hours {a.max_hours} reached")
    except KeyboardInterrupt:
        save("latest")
        log.close("stopped", note="interrupted")
    except Exception as e:
        log.close("failed", note=f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
