"""Next-Concept Modeling (NCM) baseline: a transformer trained from scratch that writes the reply couplet
coarse → fine, as a hierarchy of discrete "concept" codes (Kumbong et al., "Next-Concept Modeling: Fast, Parallel
Coarse-to-Fine Text Generation", arXiv 2610.14162; model code from third_party/ncm, MIT, which ships no trainer).

How NCM is mapped onto call → reply:
  - pre-tokenizer: OUR syllable tokenizer (data/old/processed/train_v3/tokenizer.json), not their English BPE;
  - target: one reply couplet = 48 syllable tokens (…<nl>…<eos>, right-padded with <pad>), like their XSum setup
    (L = 48, 9 levels of 4, 6, 10, 14, 20, 26, 32, 40, 48 codes);
  - condition: the rival's call (<bos> … <sep><duel|cont>, ≤ 64 tokens) is a raw-token prefix — embedded with the
    frozen tokenizer's token table, projected, and attended by every level (prefix-LM mask, as for XSum/WMT14).
    Plain couplets and dialect text get the empty condition [<sep>] (an all-pad prefix would be a fully masked row).

Two stages:
  1. tokenizer — multiscale residual VQ-VAE (EMA codebooks, nested "fine-drop", β = 0.25) on reply/couplet
     sequences and 48-token Cypriot dialect-text windows;
  2. generator — frozen tokenizer; next-concept transformer trained on all levels in one pass (block-diagonal
     attention), level-weighted code cross-entropy (w_k ∝ ℓ_k^-α), uniform code corruption of the conditioning
     levels (mean 0.5, decreasing with level). Curriculum as for the fly runs: pretrain (dialect text, no call) →
     couplets (no call) → pairs (call → reply, duel share 0.4, dialect-text replay 0.3).

Deviations from their XSum recipe (forced by one 12 GB GPU and ~4 h; see the comments at each):
  - codebooks are k-means-initialised per level and dead codes re-seeded (restart_threshold) — at batch 256 instead
    of 4096 their zero-initialised EMA collapses every level to one code;
  - untied per-level linear output heads instead of codebook-tied heads (tied heads stayed near-uniform);
  - generator 12 × 512 (≈57M incl. heads) instead of 12 × 768, effective batch 256 (16 × 16 accumulation);
  - unstated in the paper, chosen here: α = 0.5, corruption 0.75 → 0.25 by level, sampling T = 0.8 / top-k 40.

Evaluation is comparable with train.py / llm_finetune.py where NCM allows it: NCM has no exact likelihood, so its
losses are CODE-level (nats per concept code), not per syllable; context gain = code loss with a mismatched call
minus with the true call (same mismatch rule as train.py). Reply quality uses generate.py's score_reply / copy_ratio /
rank: verse = best of 16 samples (no meter-constrained decoding is possible — NCM emits codes, not syllables),
raw = 1 sample, on the first 16 held-out duel calls.

  .venv-llm/bin/python -I mougia_train/ncm_train.py --run-id ncm-v3
"""
import argparse
import json
import math
import os
import random
import signal
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
NCM_DIR = os.path.join(ROOT, "third_party", "ncm")
sys.path.insert(0, ROOT)
sys.path.insert(0, NCM_DIR)  # their packages are called models/ and utils/ — must win over anything in ROOT
from omegaconf import OmegaConf  # noqa: E402
from models.next_concept import NextConceptTransformer  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402

from mougia_train.generate import copy_ratio, prompt_ids, rank, score_reply  # noqa: E402
from mougia_train.runlog import RunLogger  # noqa: E402
from mougia_train.tokenizer import EOS, PAD, SEP, Tokenizer  # noqa: E402

LEVELS = [4, 6, 10, 14, 20, 26, 32, 40, 48]  # their XSum schedule (L = 48)
L = LEVELS[-1]
NULL_PREFIX = [SEP]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="ncm-v3")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "old", "processed", "train_v3"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix-len", type=int, default=64, help="call tokens (longest call in v3 is 64)")
    # stage 1: tokenizer (their XSum tokenizer config; batch/steps scaled to one GPU)
    ap.add_argument("--tok-dim", type=int, default=384)
    ap.add_argument("--tok-codebook", type=int, default=4096)
    ap.add_argument("--tok-batch", type=int, default=256)
    ap.add_argument("--tok-steps", type=int, default=15000)
    ap.add_argument("--tok-lr", type=float, default=5e-4)
    ap.add_argument("--tok-warmup", type=int, default=1000)
    ap.add_argument("--tok-dropout", type=float, default=0.1)
    ap.add_argument("--tok-ema", type=float, default=0.99)
    ap.add_argument("--tok-restart", type=float, default=0.01,
                    help="re-seed codes whose EMA usage falls below this (their restart_threshold option)")
    ap.add_argument("--tok-kmeans-batches", type=int, default=32, help="batches for the k-means codebook init")
    ap.add_argument("--tokenizer-ckpt", default=None, help="skip stage 1 and load this tokenizer checkpoint")
    # stage 2: generator
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.3, help="their XSum generator dropout")
    ap.add_argument("--batch", type=int, default=16, help="micro-batch (≈2.7 GB peak at 12×512)")
    ap.add_argument("--accum", type=int, default=16, help="gradient accumulation (effective batch = batch × accum)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ft-lr", type=float, default=1e-4, help="peak LR of the couplets / pairs phases")
    ap.add_argument("--min-lr-frac", type=float, default=0.1, help="LR floor = 10%% of peak, as in the paper")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.5, help="level weighting w_k ∝ ℓ_k^-α (not stated in the paper)")
    ap.add_argument("--corrupt-hi", type=float, default=0.75, help="code-corruption prob. at the coarsest level")
    ap.add_argument("--corrupt-lo", type=float, default=0.25, help="… at the finest conditioning level (mean 0.5)")
    ap.add_argument("--pretrain-steps", type=int, default=8000)
    ap.add_argument("--couplet-steps", type=int, default=1200)
    ap.add_argument("--pair-steps", type=int, default=1500)
    ap.add_argument("--duel-share", type=float, default=0.4)
    ap.add_argument("--lm-replay", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.8, help="sampling temperature (same as the fly / Krikri runs)")
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--n-verse", type=int, default=16)
    ap.add_argument("--eval-replies", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every-min", type=float, default=10)
    ap.add_argument("--ft-eval-every-min", type=float, default=5, help="eval interval in the couplets / pairs phases (early stopping)")
    ap.add_argument("--sample-every-min", type=float, default=15)
    ap.add_argument("--ckpt-every-min", type=float, default=30)
    ap.add_argument("--max-hours", type=float, default=0)
    return ap.parse_args()


# ------------------------------------------------------------------------------------------------- data
class Data:
    def __init__(self, a):
        d = a.data
        self.a = a
        self.P = a.prefix_len
        self.lm_train = torch.load(os.path.join(d, "lm_train.pt")).long()
        self.lm_val = torch.load(os.path.join(d, "lm_val.pt")).long()
        strip = lambda c: c[1:] if c and c[0] == 2 else c  # couplets start with <bos>; replies don't
        self.c_train = [strip(c) for c in torch.load(os.path.join(d, "couplets_train.pt")) if len(strip(c)) <= L]
        # validation keeps every example (truncated to L if longer) so the sets match train.py's
        self.c_val = [strip(c) for c in torch.load(os.path.join(d, "couplets_val.pt"))][:256]
        self.p_train = [p for p in torch.load(os.path.join(d, "pairs_train.pt")) if len(p["reply"]) <= L]
        self.p_val = torch.load(os.path.join(d, "pairs_val.pt"))
        self.duel = [p for p in self.p_train if p["style"] == "duel"]
        self.cont = [p for p in self.p_train if p["style"] == "cont"]
        self.duel_w = [p["w"] for p in self.duel]
        n = len(self.lm_val) - L
        self.lm_val_windows = [self.lm_val[s:s + L].tolist() for s in range(0, n, max(1, n // 256))][:256]

    # tensors ----------------------------------------------------------------------------------
    @staticmethod
    def targets(seqs):
        out = torch.full((len(seqs), L), PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            s = list(s)[:L]
            out[i, :len(s)] = torch.tensor(s)
        return out

    def prefixes(self, calls):
        out = torch.full((len(calls), self.P), PAD, dtype=torch.long)
        for i, c in enumerate(calls):
            c = list(c)[-self.P:]  # keep the end (<sep><duel>) if a free-form prompt is longer
            out[i, :len(c)] = torch.tensor(c)
        return out

    def windows(self, n):
        starts = torch.randint(0, len(self.lm_train) - L, (n,))
        return [self.lm_train[s:s + L].tolist() for s in starts]

    # batches ----------------------------------------------------------------------------------
    def tok_batch(self, B):
        """Tokenizer training: ⅓ dialect-text windows, ⅓ couplets, ⅓ replies."""
        k = B // 3
        seqs = self.windows(B - 2 * k) + random.sample(self.c_train, k) + [p["reply"] for p in random.sample(self.p_train, k)]
        return self.targets(seqs)

    def gen_batch(self, kind, B):
        if kind == "lm":
            return self.prefixes([NULL_PREFIX] * B), self.targets(self.windows(B))
        if kind == "couplets":
            return self.prefixes([NULL_PREFIX] * B), self.targets(random.sample(self.c_train, B))
        k = sum(random.random() < self.a.duel_share for _ in range(B))
        pick = random.choices(self.duel, weights=self.duel_w, k=k) + random.choices(self.cont, k=B - k)
        return self.prefixes([p["call"] for p in pick]), self.targets([p["reply"] for p in pick])


# ------------------------------------------------------------------------------------------------- models
def tokenizer_config(a, V):
    return dict(
        in_vocab_size=V, context_length=L, embed_dim=a.tok_dim, vqvae_vocab_size=[a.tok_codebook] * len(LEVELS),
        n_layers=6,  # = decoder attention heads in their code
        compression_factor=1, dropout=a.tok_dropout, pre_quant_groupnorm=8, use_qk_norm=True, pre_quant_dropout=0.25,
        enc_layers=0, dec_layers=1,
        vector_quantizer_config=dict(
            clss="multiscale_residual_vector_quantizer", decay=a.tok_ema, epsilon=1e-5, commitment_cost=0.25,
            restart_threshold=a.tok_restart,
            learned_l1_sampling=True, learned_l1_cascade=True, learned_l1_cascade_base=4, learned_all_sampling=False,
            learned_l1_cascade_norm=True, learned_l1_cascade_norm_final=True,
            quant_resi=dict(enabled=True, ratio=0.5, share_mode=0, learnable_ratio=False),
            aux=dict(fine_drop=dict(prob=0.5, min_keep=1)),
            levels=dict(use_manual_levels=True, manual_levels=LEVELS)))


def build_tokenizer(cfg):
    cfg = dict(cfg)
    cfg["vector_quantizer_config"] = OmegaConf.create(cfg["vector_quantizer_config"])
    return VQVAE(**cfg)


@torch.no_grad()
def kmeans_init(vq, data, a, dev):
    """Seed every level's codebook by k-means on that level's residuals (their kmeans_init_level, which their released
    code never calls). Without it the EMA codebooks collapse at our batch size: ema_cluster_size starts at 0, so every
    code not hit in the very first batch is divided by ε and never chosen again (their batch of 4096 hides this)."""
    was = vq.training
    vq.eval()
    q = vq.vq
    z = torch.cat([vq._encode_pre_quant(data.tok_batch(a.tok_batch).to(dev)) for _ in range(a.tok_kmeans_batches)]).float()
    B, Lz, C = z.shape
    for i, lv in enumerate(LEVELS):
        last = i == len(LEVELS) - 1
        flat = (z if last else q._down_for_quant(z, i, lv)).reshape(-1, C)
        pick = flat[torch.randperm(len(flat), device=flat.device)[:65536]]
        q.kmeans_init_level(i, pick, n_iters=10)
        E = q.embeddings[i].weight
        idx = torch.cat([q._distances(flat[j:j + 8192], E).argmin(1) for j in range(0, len(flat), 8192)]).view(B, lv)
        quant = q.embeddings[i](idx)
        quant = quant if last else q._up_to_full(quant, i, Lz)
        z = z - q._apply_quant_resi(quant, i)
    vq.train(was)


def generator_config(a, tok_cfg):
    return dict(n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd, bias=True, dropout=a.dropout,
                prefix_len=a.prefix_len, raw_token_prefix=True, prefix_pad_token_id=PAD,
                vocab_size=list(tok_cfg["vqvae_vocab_size"]), predict_level0=False, query_level0=True,
                attn_pattern="block_diagonal", use_level_encoding=True, use_rope=True, rope_base=100000,
                use_qk_norm=True, use_flex_attention=False,
                # Their XSum config ties each head to its codebook (distance logits × one learned temperature). On our
                # budget that head stays near-uniform (the temperature moves ~1e-4/step; code loss ≈ ln 4096 after
                # 900 steps), so we use their untied per-level linear heads (tie_heads_to_codebook: false).
                tie_heads_to_codebook=False, untied_head_bias=True)


def lr_scale(k, total, warmup, floor):
    return min(1.0, (k + 1) / warmup) * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, k / max(1, total)))))


# ------------------------------------------------------------------------------------------------- main
def _stop(*_):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    a = parse()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    amp = dict(device_type="cuda", dtype=torch.bfloat16)

    tok = Tokenizer.load(os.path.join(a.data, "tokenizer.json"))
    data = Data(a)
    prompts = json.load(open(os.path.join(a.data, "sample_prompts.json"), encoding="utf-8"))
    tok_cfg = tokenizer_config(a, len(tok))
    gen_cfg = generator_config(a, tok_cfg)

    vq = build_tokenizer(tok_cfg).to(dev)
    if a.tokenizer_ckpt:
        vq.load_state_dict(torch.load(a.tokenizer_ckpt, map_location=dev, weights_only=False)["model"])
    # A throw-away generator just for the parameter count in config.json (the real one is built after stage 1,
    # because its output heads are tied to the trained codebooks).
    n_gen = sum(p.numel() for p in NextConceptTransformer(vq, SimpleNamespace(**gen_cfg)).parameters())
    n_tok = sum(p.numel() for p in vq.parameters())

    log = RunLogger(a.run_id, {
        "label": "NCM (next-concept transformer, from scratch)", "poet": "NCM", "kind": "ncm_baseline", "args": vars(a),
        "paper": "Kumbong et al., Next-Concept Modeling, arXiv 2610.14162 (code: third_party/ncm, MIT)",
        "params": {"generator": n_gen, "tokenizer": n_tok, "total": n_gen + n_tok},
        "tokenizer": {k: v for k, v in tok_cfg.items()}, "generator": gen_cfg, "levels": LEVELS, "target_len": L,
        "data": {"couplets_train": len(data.c_train), "pairs_train": len(data.p_train), "duel_train": len(data.duel),
                 "cont_train": len(data.cont), "lm_train_tokens": len(data.lm_train), "pairs_val": len(data.p_val)},
        "data_dir": a.data, "vocab": len(tok), "sample_prompts": [p["id"] for p in prompts],
        "note": ("Losses are CODE-level: nats per concept code (level-weighted, w_k ∝ ℓ_k^-α), not per syllable — NCM has "
                 "no exact token likelihood, so val losses / context gain are not directly comparable with the fly or "
                 "Krikri per-syllable numbers; compare the generation metrics. context_gain_* = code loss with a "
                 "mismatched call (train.py's rule) minus with the true call. Verse mode = best of 16 unconstrained "
                 "samples ranked by generate.rank (logp = mean code log-prob); NCM cannot use meter-constrained "
                 "decoding. Tokenizer codebooks are k-means initialised + dead-code restarts, and the generator uses untied "
                 "per-level heads (deviations from the XSum recipe, see ncm_train.py). Replies longer than 48 syllable tokens (0.3 percent of train) are dropped from training and "
                 "truncated in validation."),
    }, sample_every_s=a.sample_every_min * 60)

    step = 0
    phase = "tokenizer"
    gen = None
    best = {}

    def save(obj, name):
        torch.save(obj, os.path.join(log.dir, "checkpoints", name))

    # ===================================================================== stage 1: tokenizer
    val_tok = torch.cat([data.targets(data.c_val), data.targets([p["reply"] for p in data.p_val]),
                         data.targets(data.lm_val_windows)])

    @torch.no_grad()
    def tok_eval():
        vq.eval()
        out, tot, acc, acc_np, n_np, exact, used = {}, 0.0, 0, 0, 0, 0, [set() for _ in LEVELS]
        for i in range(0, len(val_tok), 256):
            x = val_tok[i:i + 256].to(dev)
            idx = vq.encode(x)
            off = 0
            for k, lv in enumerate(LEVELS):
                used[k].update(idx[:, off:off + lv].flatten().tolist())
                off += lv
            with torch.autocast(**amp):
                logits = vq.decode(idx).float()
            tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x.reshape(-1), reduction="sum").item()
            pred = logits.argmax(-1)
            acc += (pred == x).sum().item()
            m = x != PAD
            acc_np += ((pred == x) & m).sum().item()
            n_np += m.sum().item()
            exact += (pred == x).all(1).sum().item()
        out["tok_val_loss"] = tot / val_tok.numel()
        out["tok_val_token_acc"] = acc / val_tok.numel()
        out["tok_val_syllable_acc"] = acc_np / max(1, n_np)  # non-pad positions only
        out["tok_val_exact"] = exact / len(val_tok)
        for k, u in enumerate(used):
            out[f"tok_codes_used_l{k}"] = len(u) / a.tok_codebook
        # what the coarse levels alone carry: reconstruct from the first k levels (k = 1, 3, 5, 7)
        x = val_tok[:256].to(dev)
        # (their idx_to_zhats accumulates in place, so its per-level list is all the final z_hat: sum it here)
        idx, q, zh, off = vq.encode(x), vq.vq, 0, 0
        cum = []
        for i, lv in enumerate(LEVELS):
            e = q.embeddings[i](idx[:, off:off + lv])
            zh = zh + q._apply_quant_resi(e if i == len(LEVELS) - 1 else q._up_to_full(e, i, L), i)
            cum.append(zh)
            off += lv
        m = x != PAD
        for k in (1, 3, 5, 7):
            with torch.autocast(**amp):
                pred = vq.decoder(cum[k - 1]).float().argmax(-1)
            out[f"tok_val_syllable_acc_first{k}"] = ((pred == x) & m).sum().item() / max(1, m.sum().item())
        vq.train()
        return out

    try:
        if not a.tokenizer_ckpt:
            kmeans_init(vq, data, a, dev)
            vq.train()
            params = [p for p in vq.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=a.tok_lr, betas=(0.9, 0.95), weight_decay=0.01)
            t_last, last_eval, best_tok = time.time(), log.elapsed(), float("inf")
            for k in range(a.tok_steps):
                for g in opt.param_groups:
                    g["lr"] = a.tok_lr * lr_scale(k, a.tok_steps, a.tok_warmup, 0.1)
                x = data.tok_batch(a.tok_batch).to(dev)
                z = vq._encode_pre_quant(x)
                z_q, vq_loss, _, _, _ = vq.vq(z)  # fp32: nearest-code search must not run in bf16
                with torch.autocast(**amp):
                    logits = vq.decoder(z_q)
                recon = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x.reshape(-1))
                loss = recon + vq_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                step += 1
                if step % a.log_every == 0:
                    torch.cuda.synchronize()
                    dt = time.time() - t_last
                    log.metrics(step, phase, train_loss=loss.item(), tok_train_loss=loss.item(), tok_recon_loss=recon.item(),
                                tok_vq_loss=float(vq_loss), tok_train_acc=(logits.argmax(-1) == x).float().mean().item(),
                                lr=opt.param_groups[0]["lr"], grad_norm=float(gn),
                                tokens_per_s=a.log_every * x.numel() / max(dt, 1e-6),
                                gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9, phase_progress=(k + 1) / a.tok_steps)
                    t_last = time.time()
                if log.elapsed() - last_eval >= a.eval_every_min * 60 or k == a.tok_steps - 1:
                    with log.sampling():
                        res = tok_eval()
                        log.eval(step, phase, **res)
                        if res["tok_val_loss"] < best_tok:
                            best_tok = res["tok_val_loss"]
                            save({"model": vq.state_dict(), "config": tok_cfg, "step": step, "val": res}, "tokenizer.pt")
                    last_eval = log.elapsed()
                    t_last = time.time()
                if a.max_hours and log.elapsed() > a.max_hours * 3600:
                    raise TimeoutError
            save({"model": vq.state_dict(), "config": tok_cfg, "step": step}, "tokenizer-end.pt")
            vq.load_state_dict(torch.load(os.path.join(log.dir, "checkpoints", "tokenizer.pt"), weights_only=False)["model"])
            del opt
        else:
            with log.sampling():
                log.eval(step, phase, **tok_eval())
            save({"model": vq.state_dict(), "config": tok_cfg, "step": step, "from": a.tokenizer_ckpt}, "tokenizer.pt")
        vq.eval()
        for p in vq.parameters():
            p.requires_grad_(False)
        torch.cuda.empty_cache()

        # ===================================================================== stage 2: generator
        gen = NextConceptTransformer(vq, SimpleNamespace(**gen_cfg)).to(dev)
        K = len(LEVELS)
        lvl = torch.cat([torch.full((lv,), k) for k, lv in enumerate(LEVELS)]).to(dev)
        lens = torch.tensor(LEVELS, dtype=torch.float)
        w_code = (lens[lvl.cpu()] ** -a.alpha / (lens ** (1 - a.alpha)).sum()).to(dev)  # Σ over codes = 1
        corrupt_p = torch.linspace(a.corrupt_hi, a.corrupt_lo, K - 1)
        corrupt_code_p = torch.cat([torch.full((lv,), float(corrupt_p[k])) for k, lv in enumerate(LEVELS[:-1])]
                                   + [torch.zeros(LEVELS[-1])]).to(dev)

        def gen_logits(prefix, idx, corrupt):
            """Teacher-forced pass over all levels. idx = clean target codes (B, Σℓ)."""
            with torch.no_grad():
                idx_in = idx
                if corrupt:
                    m = torch.rand(idx.shape, device=dev) < corrupt_code_p
                    idx_in = torch.where(m, torch.randint_like(idx, a.tok_codebook), idx)
                ncm_in = vq.vq.idx_to_ncm_input(idx_in)
                zp = vq.encoder.wte(prefix)
            B = idx.shape[0]
            x = torch.cat([zp, gen.l0_queries.expand(B, -1, -1).to(zp.dtype), ncm_in], dim=1)
            with torch.autocast(**amp):
                return gen(x, prefix_pad_mask=prefix == PAD).float()

        def code_loss(logits, idx, per_example=False):
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1), reduction="none").view(idx.shape)
            per = (ce * w_code).sum(1)
            return per if per_example else per.mean()

        @torch.no_grad()
        def loss_on(prefix, tgt):
            tot, n = 0.0, 0
            for i in range(0, len(tgt), 128):
                p, t = prefix[i:i + 128].to(dev), tgt[i:i + 128].to(dev)
                idx = vq.encode(t)
                tot += code_loss(gen_logits(p, idx, corrupt=False), idx, per_example=True).sum().item()
                n += len(t)
            return tot / max(1, n)

        @torch.no_grad()
        def sample_replies(calls, n):
            """calls: list of call-id lists. Returns, per call, n candidate dicts (reply text, mean code logp)."""
            gen.eval()
            pref = data.prefixes([c for c in calls for _ in range(n)]).to(dev)
            outs = []
            for i in range(0, len(pref), 128):
                p = pref[i:i + 128]
                with torch.autocast(**amp):
                    idx = gen.generate(None, p, temperature=a.temperature, top_k=a.top_k, top_p=1.0)
                lp = torch.log_softmax(gen_logits(p, idx, corrupt=False), -1).gather(-1, idx[..., None]).squeeze(-1).mean(1)
                zh = vq.vq.idx_to_zhats(idx, return_zlevels=False)[0][-1]
                with torch.autocast(**amp):
                    ids = vq.decoder(zh).float().argmax(-1).tolist()
                for row, l in zip(ids, lp.tolist()):
                    row = row[:row.index(EOS)] if EOS in row else row
                    outs.append((tok.decode([t for t in row if t != PAD]), l))
            gen.train()
            return [outs[j * n:(j + 1) * n] for j in range(len(calls))]

        def reply_for(calls, call_texts, mode):
            n = 1 if mode == "raw" else a.n_verse
            res = []
            for cands, text in zip(sample_replies(calls, n), call_texts):
                cs = [{"reply": r, "logp": round(l, 3), "scores": score_reply(r), "copy": round(copy_ratio(r, text), 2)} for r, l in cands]
                cs.sort(key=rank, reverse=True)
                res.append(cs[0])
            return res

        duel_val = [p for p in data.p_val if p["style"] == "duel"]
        cont_val = [p for p in data.p_val if p["style"] == "cont"][:64]

        def other_calls(ps):  # train.py's rule: far away in the list, never the pair's own reply
            out = []
            for i in range(len(ps)):
                j = (i + len(ps) // 2) % len(ps)
                while ps[j]["call"][1:-2] == ps[i]["reply"][:-1] or j == i:
                    j = (j + 1) % len(ps)
                out.append(ps[j]["call"])
            return out

        null = lambda n: data.prefixes([NULL_PREFIX] * n)

        def evaluate(ph):
            gen.eval()
            out = {"val_loss_lm": loss_on(null(len(data.lm_val_windows)), data.targets(data.lm_val_windows)),
                   "val_loss_couplets": loss_on(null(len(data.c_val)), data.targets(data.c_val))}
            for style, ps in (("duel", duel_val), ("cont", cont_val)):
                tgt = data.targets([p["reply"] for p in ps])
                true = loss_on(data.prefixes([p["call"] for p in ps]), tgt)
                out[f"val_loss_{style}"] = true
                out[f"context_gain_{style}"] = loss_on(data.prefixes(other_calls(ps)), tgt) - true
            out["val_loss"] = out[{"pretrain": "val_loss_lm", "couplets": "val_loss_couplets", "pairs": "val_loss_duel"}[ph]]
            out["val_ppl"] = math.exp(min(out["val_loss"], 20))  # per-code perplexity
            calls = [p["call"] for p in duel_val[: a.eval_replies]]
            texts = [tok.decode(c) for c in calls]
            for mode, suffix in (("raw", "_raw"), ("verse", "")):
                best_c = reply_for(calls, texts, mode)
                sc = [c["scores"] for c in best_c]
                out["meter_rate" + suffix] = sum(s["meter_ok"] for s in sc) / len(sc)
                out["rhyme_rate" + suffix] = sum(s["rhyme_ok"] for s in sc) / len(sc)
                ok = [s for s in sc if len(s["syllables"]) == 2]
                out["cy_rate" + suffix] = sum(s["cy_score"] >= 1.5 for s in ok) / len(sc)
                out["two_line_rate" + suffix] = len(ok) / len(sc)
                out["copy_rate" + suffix] = sum(c["copy"] >= 0.6 for c in best_c) / len(sc)
            gen.train()
            return out

        def sample(ph):
            calls = [prompt_ids(tok, p["text"], "duel") for p in prompts]
            verse = reply_for(calls, [p["text"] for p in prompts], "verse")
            raw = reply_for(calls, [p["text"] for p in prompts], "raw")
            for p, v, r in zip(prompts, verse, raw):
                log.sample(step, ph, prompt=p["text"], reply=v["reply"], prompt_id=p["id"], true_reply=p.get("true_reply"),
                           raw_reply=r["reply"], mode="verse (best of 16, unconstrained)", logp=v["logp"], **v["scores"])

        def save_gen(tag):
            save({"model": gen.state_dict(), "config": gen_cfg, "tokenizer_config": tok_cfg, "step": step, "args": vars(a)},
                 f"{tag}.pt")

        decay, no_decay = [], []
        for n_, p in gen.named_parameters():
            (decay if p.dim() >= 2 and "wte" not in n_ and "wle" not in n_ else no_decay).append(p)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                                lr=a.lr, betas=(0.9, 0.95))
        gen.train()
        last_eval, last_ckpt = -1e18, log.elapsed()
        phases = [("pretrain", a.pretrain_steps, a.lr), ("couplets", a.couplet_steps, a.ft_lr), ("pairs", a.pair_steps, a.ft_lr)]
        for phase, n_steps, peak in phases:
            if n_steps <= 0:
                continue
            bad, t_last, seen = 0, time.time(), 0
            for k in range(n_steps):
                for g in opt.param_groups:
                    g["lr"] = peak * lr_scale(k, n_steps, a.warmup, a.min_lr_frac)
                opt.zero_grad(set_to_none=True)
                loss_acc, n_replay = 0.0, 0
                for _ in range(a.accum):
                    replay = phase != "pretrain" and random.random() < a.lm_replay
                    n_replay += replay
                    kind = "lm" if (phase == "pretrain" or replay) else phase
                    prefix, tgt = data.gen_batch(kind, a.batch)
                    prefix, tgt = prefix.to(dev), tgt.to(dev)
                    idx = vq.encode(tgt)
                    logits = gen_logits(prefix, idx, corrupt=True)
                    loss = code_loss(logits, idx)
                    (loss / a.accum).backward()
                    loss_acc += loss.item() / a.accum
                    seen += len(tgt)
                gn = torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
                if not torch.isfinite(gn):
                    log.metrics(step, phase, event="nonfinite_grad")
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                step += 1
                if step % a.log_every == 0:
                    torch.cuda.synchronize()
                    dt = time.time() - t_last
                    with torch.no_grad():
                        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1), reduction="none").view(idx.shape)
                        per_level = {f"gen_loss_l{j}": ce[:, lvl == j].mean().item() for j in (0, 2, 4, 6, 8)}
                    log.metrics(step, phase, train_loss=loss_acc, gen_train_loss=loss_acc, replay_share=n_replay / a.accum, **per_level,
                                lr=opt.param_groups[0]["lr"], grad_norm=float(gn),
                                tokens_per_s=seen * L / max(dt, 1e-6), codes_per_s=seen * idx.shape[1] / max(dt, 1e-6),
                                gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9, phase_progress=(k + 1) / n_steps)
                    t_last, seen = time.time(), 0
                if log.elapsed() - last_eval >= (a.eval_every_min if phase == "pretrain" else a.ft_eval_every_min) * 60:
                    with log.sampling():
                        res = evaluate(phase)
                        log.eval(step, phase, **res)
                        improved = res["val_loss"] < best.get(phase, float("inf"))
                        if improved:
                            best[phase] = res["val_loss"]
                            save_gen(f"best-{phase}")
                    last_eval = log.elapsed()
                    bad = 0 if improved else bad + 1
                    t_last, seen = time.time(), 0
                    if phase != "pretrain" and bad >= a.patience:
                        log.metrics(step, phase, event=f"early_stop after {bad} evals without improvement")
                        break
                if log.should_sample():
                    with log.sampling():
                        sample(phase)
                    t_last, seen = time.time(), 0
                if log.elapsed() - last_ckpt >= a.ckpt_every_min * 60:
                    save_gen("latest")
                    last_ckpt = log.elapsed()
                if a.max_hours and log.elapsed() > a.max_hours * 3600:
                    raise TimeoutError
            with log.sampling():
                res = evaluate(phase)
                log.eval(step, phase, **res)
                if res["val_loss"] < best.get(phase, float("inf")):
                    best[phase] = res["val_loss"]
                    save_gen(f"best-{phase}")
                sample(phase)
            last_eval = log.elapsed()
            save_gen(f"end-{phase}")
            if phase != "pretrain" and phase in best:  # next phase starts from this phase's best weights
                gen.load_state_dict(torch.load(os.path.join(log.dir, "checkpoints", f"best-{phase}.pt"), weights_only=False)["model"])
        log.close("finished")
    except TimeoutError:
        if gen is not None:
            with log.sampling():
                log.eval(step, phase, **evaluate(phase))
                sample(phase)
            save_gen("latest")
        log.close("stopped", note=f"max-hours {a.max_hours} reached")
    except KeyboardInterrupt:
        if gen is not None:
            save_gen("latest")
        log.close("stopped", note="interrupted")
    except Exception as e:
        log.close("failed", note=f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
