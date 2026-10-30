"""Train a fly-brain ποιητάρης (Kyriakou = female FlyWire, Markos = male CNS) in three phases:

  1. pretrain  — next-syllable prediction on Cypriot dialect text
  2. couplets  — next-syllable prediction on tsiattista / folk couplets
  3. pairs     — answer a call couplet with a reply couplet (loss on the reply only)

Everything is logged through RunLogger to runs/<run_id>/ (see runlog.py) and appears live in the browser's
Training tab: per-step metrics, periodic evaluations (val loss + meter / rhyme / Cypriot rates of generated replies),
reply samples every 15 minutes of training time, and checkpoints.

Example:
  python3 -I mougia_train/train.py --sex female --region central --run-id pilot-kyriakou-central
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mougia_train.generate import BAN, prompt_ids, reply as gen_reply, score_reply, verse_masks  # noqa: E402
from mougia_train.model import FlyBrainLM, load_graph, load_mb  # noqa: E402
from mougia_train.poets import POETS, run_label  # noqa: E402
from mougia_train.runlog import RunLogger  # noqa: E402
from mougia_train.tokenizer import BOS, EOS, PAD, SEP, UNK, Tokenizer  # noqa: E402

DATA = os.path.join(ROOT, "data", "old", "processed", "train")
GRAPHS = os.path.join(ROOT, "data", "old", "connectomes", "processed")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sex", choices=["female", "male"], required=True)
    ap.add_argument("--region", choices=["full", "central"], default="full")
    ap.add_argument("--wiring", choices=["connectome", "random"], default="connectome")
    ap.add_argument("--max-edges", type=int, default=0, help="keep only the strongest N connections (density matching)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--data", default=None, help="training-data folder from prepare.py (default data/old/processed/train)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ticks", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--readout-hidden", type=int, default=512)
    ap.add_argument("--no-learn-edges", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--edge-lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1, help="readout dropout")
    ap.add_argument("--no-learn-tau", action="store_true", help="fixed integration rate for every neuron")
    ap.add_argument("--interface", choices=["neuron", "type"], default="neuron",
                    help="neuron (v4): per-neuron smell/hearing inputs and per-neuron readout; type: v1–v3 per-type channels")
    ap.add_argument("--heard-split", type=float, default=0.5,
                    help="share of dialect-text windows split into a heard part and an own part with <sep> (trains the ears)")
    ap.add_argument("--no-ctx-drive", action="store_true", help="no running summary of the sequence into the input neurons")
    ap.add_argument("--lm-replay", type=float, default=0.3, help="share of dialect-text batches mixed into later phases")
    ap.add_argument("--edge-lr-after-pretrain", type=float, default=0.1, help="edge learning-rate multiplier after pretraining")
    ap.add_argument("--patience", type=int, default=3, help="evals without val improvement before a later phase stops early")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--pretrain-tokens", type=float, default=11.5e6, help="≈2 passes over the dialect text")
    ap.add_argument("--couplet-tokens", type=float, default=1.5e6, help="≈4 passes over the couplets (early stopping may end sooner)")
    ap.add_argument("--pair-tokens", type=float, default=1.5e6, help="≈7 passes over the reply tokens of all pairs (early stopping may end sooner)")
    ap.add_argument("--duel-share", type=float, default=0.4, help="fraction of reply batches drawn from real duel pairs")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every-min", type=float, default=10)
    ap.add_argument("--sample-every-min", type=float, default=15)
    ap.add_argument("--ckpt-every-min", type=float, default=30)
    ap.add_argument("--eval-replies", type=int, default=24)
    ap.add_argument("--max-hours", type=float, default=0, help="stop after this much training time (0 = no limit)")
    ap.add_argument("--mb", action="store_true", help="mushroom-body memory: plastic KC→MBON synapses gated by dopamine (v5)")
    ap.add_argument("--mb-decay", type=float, default=0.95, help="initial per-syllable retention λ of the mushroom-body memory")
    return ap.parse_args()


# ---------------------------------------------------------------------------------------------------------------
class Batches:
    def __init__(self, a, tok):
        self.a = a
        self.lm_train = torch.load(os.path.join(DATA, "lm_train.pt")).long()
        self.lm_val = torch.load(os.path.join(DATA, "lm_val.pt")).long()
        self.c_train = torch.load(os.path.join(DATA, "couplets_train.pt"))
        self.c_val = torch.load(os.path.join(DATA, "couplets_val.pt"))
        self.p_train = torch.load(os.path.join(DATA, "pairs_train.pt"))
        self.p_val = torch.load(os.path.join(DATA, "pairs_val.pt"))

    def lm(self, split="train", n=None):
        """Dialect-text windows. With probability heard_split a <sep> is inserted at a random point: the text before
        it is 'heard' (auditory input), the rest is the fly's own continuation — so the ears learn on lots of text."""
        src = self.lm_train if split == "train" else self.lm_val
        B, L = n or self.a.batch, self.a.seq_len
        starts = torch.randint(0, len(src) - L - 1, (B,))
        x = torch.stack([src[s:s + L + 1] for s in starts])
        if self.a.interface == "neuron" and split == "train" and self.a.heard_split > 0:
            for i in range(B):
                if random.random() < self.a.heard_split:
                    k = random.randint(8, L - 8)
                    x[i] = torch.cat([x[i, :k], torch.tensor([SEP]), x[i, k:L]])
        mask = x[:, 1:] != SEP  # never ask the model to predict the inserted <sep>
        return x[:, :-1], x[:, 1:], mask

    @staticmethod
    def pad(seqs, loss_from):
        L = max(len(s) for s in seqs) - 1
        inp = torch.full((len(seqs), L), PAD, dtype=torch.long)
        tgt = torch.full((len(seqs), L), PAD, dtype=torch.long)
        mask = torch.zeros(len(seqs), L, dtype=torch.bool)
        for i, (s, k) in enumerate(zip(seqs, loss_from)):
            s = torch.tensor(s[: L + 1])
            inp[i, : len(s) - 1], tgt[i, : len(s) - 1] = s[:-1], s[1:]
            mask[i, max(0, k - 1): len(s) - 1] = True
        return inp, tgt, mask

    def couplets(self, split="train", n=None):
        src = self.c_train if split == "train" else self.c_val
        seqs = random.sample(src, min(n or self.a.batch, len(src)))
        return self.pad(seqs, [1] * len(seqs))

    def pairs(self, split="train", n=None, style=None):
        """Reply batches. Training mixes real duels (duel-share of the batch, strict replies weighted ×3) with
        continuation pairs; validation can be restricted to one style."""
        if split == "train":
            duel = [p for p in self.p_train if p["style"] == "duel"]
            cont = [p for p in self.p_train if p["style"] == "cont"]
            B = n or self.a.batch
            k = sum(random.random() < self.a.duel_share for _ in range(B))
            pick = random.choices(duel, weights=[p["w"] for p in duel], k=k) + random.choices(cont, k=B - k)
        else:
            pick = [p for p in self.p_val if style in (None, p["style"])][: n or None]
        return self.pad([p["call"] + p["reply"] for p in pick], [len(p["call"]) for p in pick])


def masked_ce(logits, tgt, mask):
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none")
    m = mask.reshape(-1).float()
    return (loss * m).sum() / m.sum().clamp_min(1)



# ---------------------------------------------------------------------------------------------------------------
def _stop(*_):
    raise KeyboardInterrupt


def main():
    # Background jobs ignore SIGINT; make both SIGINT and SIGTERM stop cleanly (checkpoint + status "stopped").
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    a = parse()
    global DATA
    if a.data:
        DATA = os.path.abspath(a.data)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = "cuda"
    tok = Tokenizer.load(os.path.join(DATA, "tokenizer.json"))
    graph = load_graph(os.path.join(GRAPHS, f"{a.sex}_{a.region}.pt"), wiring=a.wiring, max_edges=a.max_edges or None, seed=a.seed)
    mb = load_mb(os.path.join(GRAPHS, f"{a.sex}_mb.pt"), a.region, graph["n"], wiring=a.wiring, seed=a.seed) if a.mb else None
    model = FlyBrainLM(graph, len(tok), d_model=a.d_model, ticks=a.ticks, alpha=a.alpha,
                       learn_edges=not a.no_learn_edges, readout_hidden=a.readout_hidden, learn_tau=not a.no_learn_tau,
                       ctx_drive=not a.no_ctx_drive and a.interface == "type", dropout=a.dropout,
                       interface=a.interface, sep_id=SEP, mb=mb, mb_decay=a.mb_decay).to(dev)
    data = Batches(a, tok)
    masks = verse_masks(tok)
    prompts = json.load(open(os.path.join(DATA, "sample_prompts.json"), encoding="utf-8"))
    data_stats = json.load(open(os.path.join(DATA, "stats.json")))

    label = run_label(a.sex, a.wiring) + ("" if a.region == "full" else f" · {a.region} brain") + (f" · top {a.max_edges:,} edges" if a.max_edges else "") + (" · mushroom-body memory" if a.mb else "")
    run_id = a.run_id or f"{POETS[a.sex]['name'].lower()}-{a.region}-{a.wiring}-{time.strftime('%Y%m%d-%H%M')}"
    params = model.n_params()
    log = RunLogger(run_id, {
        "label": label, "poet": POETS[a.sex]["name"], "sex": a.sex, "connectome": POETS[a.sex]["connectome"], "args": vars(a),
        "graph": {"neurons": graph["n"], "edges": int(len(graph["pre"])), "synapses": int(graph["syn"].sum()),
                  "interface": a.interface, "input_neurons": int(len(model.in_idx)),
                  "audio_neurons": int(len(model.au_idx)) if a.interface == "neuron" else 0, "output_channels": model.n_out,
                  "output_neurons": int(len(model.out_idx)), "min_syn": graph.get("min_syn"), "wiring": a.wiring,
                  "mb_plastic_edges": int(len(model.mb_kc)) if a.mb else 0, "mb_mbons": model.n_mbon if a.mb else 0,
                  "mb_dan_edges": int(len(model.mb_dan)) if a.mb else 0},
        "params": {"total": sum(params.values()), "edges": params.get("edge_logscale", 0), "by_tensor": params},
        "data": data_stats, "data_dir": DATA, "vocab": len(tok), "sample_prompts": [p["id"] for p in prompts],
    }, sample_every_s=a.sample_every_min * 60)

    edge_params = [model.edge_logscale] if model.edge_logscale is not None else []
    mb_params = [p for n, p in model.named_parameters() if n.startswith("mb_")]  # no decay: it would pull λ to 0.5, η to 0
    other = [p for n, p in model.named_parameters() if n != "edge_logscale" and not n.startswith("mb_")]
    opt = torch.optim.AdamW([{"params": other, "lr": a.lr, "weight_decay": a.weight_decay},
                             {"params": edge_params, "lr": a.edge_lr, "weight_decay": 0.0},
                             {"params": mb_params, "lr": a.lr, "weight_decay": 0.0}])
    base_lrs = [g["lr"] for g in opt.param_groups]

    phases = [("pretrain", a.pretrain_tokens, data.lm), ("couplets", a.couplet_tokens, data.couplets), ("pairs", a.pair_tokens, data.pairs)]
    step, last_eval, last_ckpt = 0, -1e18, log.elapsed()

    def evaluate(phase):
        model.eval()
        out = {}
        with torch.no_grad():
            for name, fn, kw in (("val_loss_lm", data.lm, {"n": 32}), ("val_loss_couplets", data.couplets, {"n": 64}),
                                 ("val_loss_duel", data.pairs, {"style": "duel"}), ("val_loss_cont", data.pairs, {"style": "cont", "n": 64})):
                inp, tgt, mask = fn("val", **kw)
                logits, _ = model(inp.to(dev), grad_ckpt=False)
                out[name] = masked_ce(logits, tgt.to(dev), mask.to(dev)).item()
            # Context gain: reply loss with someone else's call minus with the true call (nats/syllable).
            # ≈0 means the model ignores what it is answering.
            for style, n in (("duel", None), ("cont", 64)):
                ps = [p for p in data.p_val if p["style"] == style][:n]
                if len(ps) < 2:
                    continue
                losses = []
                # Mismatched call from far away in the list (another exchange). Never the pair's own reply: in a
                # duel, the next pair's call IS this pair's reply, which would leak the answer.
                def other(i):
                    j = (i + len(ps) // 2) % len(ps)
                    while ps[j]["call"][1:-2] == ps[i]["reply"][:-1] or j == i:
                        j = (j + 1) % len(ps)
                    return ps[j]["call"]
                for calls in ([p["call"] for p in ps], [other(i) for i in range(len(ps))]):
                    inp, tgt, mask = data.pad([c + p["reply"] for c, p in zip(calls, ps)], [len(c) for c in calls])
                    logits, _ = model(inp.to(dev), grad_ckpt=False)
                    losses.append(masked_ce(logits, tgt.to(dev), mask.to(dev)).item())
                out[f"context_gain_{style}"] = losses[1] - losses[0]
        key = {"pretrain": "val_loss_lm", "couplets": "val_loss_couplets", "pairs": "val_loss_duel"}[phase]
        out["val_loss"] = out[key]
        out["val_ppl"] = math.exp(min(out["val_loss"], 20))
        # Reply quality on held-out duel calls: raw (one unconstrained sample) and verse mode (meter-constrained,
        # best of 8 by rhyme/meter/dialect) — the paper reports both, so decoding help is separable from learning.
        calls = [tok.decode(p["call"]) for p in data.p_val if p["style"] == "duel"][: a.eval_replies]
        for mode, suffix in (("raw", "_raw"), ("verse", "")):
            scores = [gen_reply(model, tok, c, mode=mode, n=8, masks=masks)["scores"] for c in calls]
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
            v = gen_reply(model, tok, p["text"], mode="verse", n=16, masks=masks)
            raw = gen_reply(model, tok, p["text"], mode="raw")
            log.sample(step, phase, prompt=p["text"], reply=v["reply"], prompt_id=p["id"], true_reply=p.get("true_reply"),
                       raw_reply=raw["reply"], mode="verse (best of 16)", **v["scores"])
        model.train()

    def save(tag):
        torch.save({"model": model.state_dict(), "step": step, "args": vars(a)}, os.path.join(log.dir, "checkpoints", f"{tag}.pt"))

    model.train()
    try:
        best = {}

        def eval_and_track(phase):
            """Evaluate, keep the best checkpoint per phase (by its val loss); returns True when it improved."""
            res = evaluate(phase)
            log.eval(step, phase, **res)
            if res["val_loss"] < best.get(phase, float("inf")):
                best[phase] = res["val_loss"]
                save(f"best-{phase}")
                return True
            return False

        for phase, budget, batch_fn in phases:
            seen, phase_start, t_last, tok_last, bad = 0, step, time.time(), 0, 0
            total_steps = None
            if phase != "pretrain":  # slow down synapse learning once the language is in place
                base_lrs[1] = a.edge_lr * a.edge_lr_after_pretrain
            while seen < budget:
                replay = phase != "pretrain" and random.random() < a.lm_replay
                inp, tgt, mask = data.lm("train") if replay else batch_fn("train")
                n_tok = 0 if replay else int(mask.sum())
                if total_steps is None and n_tok:  # estimate steps in phase for the cosine schedule
                    total_steps = max(1, int(budget / max(1, n_tok) / (1 - (a.lm_replay if phase != "pretrain" else 0))))
                if total_steps is None:  # estimate steps in phase for the cosine schedule
                    total_steps = max(1, int(budget / max(1, n_tok)))
                k = step - phase_start
                scale = min(1.0, (k + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, k / (total_steps or 1)))))
                for g, lr in zip(opt.param_groups, base_lrs):
                    g["lr"] = lr * scale
                logits, _ = model(inp.to(dev))
                loss = masked_ce(logits, tgt.to(dev), mask.to(dev))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                if not torch.isfinite(gn):
                    log.metrics(step, phase, event="nonfinite_grad")
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                step += 1
                seen += n_tok
                tok_last += n_tok
                if step % a.log_every == 0:
                    torch.cuda.synchronize()
                    dt = time.time() - t_last
                    ls = model.edge_logscale
                    log.metrics(step, phase, train_loss=loss.item(), lr=opt.param_groups[0]["lr"], grad_norm=float(gn), replay=replay,
                                tau_mean=float(model.rates().mean()) if model.alpha_logit is not None else None,
                                tokens_per_s=tok_last / max(dt, 1e-6), gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                                phase_progress=seen / budget, rho=model.rho.item(), in_gain=model.in_gain.item(),
                                edge_scale_mean=float(ls.mean()) if ls is not None else None,
                                edge_scale_std=float(ls.std()) if ls is not None else None,
                                mb_decay_mean=float(torch.sigmoid(model.mb_decay_logit).mean()) if model.mb else None,
                                mb_eta_mean=float(model.mb_eta.mean()) if model.mb else None,
                                mb_eta_abs=float(model.mb_eta.abs().mean()) if model.mb else None)
                    t_last, tok_last = time.time(), 0
                if log.elapsed() - last_eval >= a.eval_every_min * 60:
                    with log.sampling():
                        improved = eval_and_track(phase)
                    last_eval = log.elapsed()
                    bad = 0 if improved else bad + 1
                    if phase != "pretrain" and bad >= a.patience:
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
            if phase != "pretrain" and phase in best:  # continue the next phase from this phase's best weights
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
