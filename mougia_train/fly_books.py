"""Train a fly-brain poet on the books corpus only (data/books/poems.jsonl), epoch by epoch.

Every poem is cut into couplets (lines 1–2, 3–4, …). An example is either
  call → reply   one couplet heard through the ears, <sep>, the next couplet of the same poem as the fly's own words
  couplet        a poem of a single couplet (the Sakellarios distichs), all own words
and an epoch is one pass over every example of the training poems. Held-out poems: the same 10% split by poem id as
the Krikri books run (mougia_train/llm_corpus.py), so the two can be compared on the same verses.

After every epoch:
  val_loss               per-syllable loss on all held-out examples
  val_loss_reply         … on the reply couplets only (given the true call)
  context_gain           loss on the replies with a call from another poem minus with the true call
  memory_gain            reply loss with the mushroom-body memory switched off minus switched on (mb only)
  replies to --prompt    best 3 of 16 verse-constrained samples, written to runs/<id>/epochs.txt and samples.jsonl
Checkpoints: best (by val_loss), latest (every epoch), end. The syllable tokenizer is built from the training poems
and saved in runs/<id>/data/, so the web app's Chat and sample_checkpoint.py can load the checkpoints.

Example:
  python3 -I mougia_train/fly_books.py --sex female --mb --run-id fly-books-kyriakou-mb
"""
import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mougia_train.books_prepare import normalize  # noqa: E402
from mougia_train.generate import reply as gen_reply, verse_masks  # noqa: E402
from mougia_train.model import FlyBrainLM, load_graph, load_mb  # noqa: E402
from mougia_train.poets import POETS  # noqa: E402
from mougia_train.runlog import RunLogger  # noqa: E402
from mougia_train.tokenizer import CONT, PAD, SEP, Tokenizer  # noqa: E402
from mougia_train.train import GRAPHS, masked_ce  # noqa: E402

CORPUS = os.path.join(ROOT, "data", "books", "poems.jsonl")
PROMPT = "Διάλεε τές να 'ν' όμορφες, Στέλιο, σαν αηδόνια,\nμεν τον βουννάς τον σπόρο σου πας τα τραχώνια."


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sex", choices=["female", "male"], default="female")
    ap.add_argument("--region", choices=["full", "central"], default="full")
    ap.add_argument("--wiring", choices=["connectome", "random"], default="connectome")
    ap.add_argument("--mb", action="store_true", help="mushroom-body memory (see model.py)")
    ap.add_argument("--mb-decay", type=float, default=0.95)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ticks", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--readout-hidden", type=int, default=512)
    ap.add_argument("--no-learn-edges", action="store_true")
    ap.add_argument("--no-learn-tau", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--edge-lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=60, help="maximum epochs (cosine schedule over this many)")
    ap.add_argument("--patience", type=int, default=8, help="stop after this many epochs without a better val_loss")
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--prompt", default=PROMPT, help="couplet answered after every epoch")
    ap.add_argument("--show", type=int, default=3, help="replies to print per epoch (best of 16)")
    ap.add_argument("--log-every", type=int, default=10)
    a = ap.parse_args()
    a.interface, a.no_ctx_drive, a.max_edges = "neuron", True, 0  # fields sample_checkpoint.load_model reads
    return a


def couplets(poem):
    ls = poem["lines"]
    return ["\n".join(ls[i:i + 2]) for i in range(0, len(ls) - 1, 2)]


def examples(poems, tok):
    """(ids, reply_start) — reply_start = index of the first reply token (0 for a lone couplet)."""
    out = []
    for p in poems:
        cs = couplets(p)
        if len(cs) == 1:
            out.append((tok.encode(cs[0], bos=True, eos=True), 0))
        for c, r in zip(cs, cs[1:]):
            call = tok.encode(c, bos=True) + [SEP, CONT]
            out.append((call + tok.encode(r, eos=True), len(call)))
    return out


def batch(exs, reply_only=False):
    L = max(len(s) for s, _ in exs) - 1
    inp = torch.full((len(exs), L), PAD, dtype=torch.long)
    tgt = torch.full((len(exs), L), PAD, dtype=torch.long)
    mask = torch.zeros(len(exs), L, dtype=torch.bool)
    for i, (s, k) in enumerate(exs):
        s = torch.tensor(s)
        inp[i, :len(s) - 1], tgt[i, :len(s) - 1] = s[:-1], s[1:]
        mask[i, (max(0, k - 1) if reply_only else 0):len(s) - 1] = True
    mask &= (tgt != SEP) & (tgt != CONT)  # the separator and style token are given, not predicted
    return inp, tgt, mask


def _stop(*_):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    a = parse()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = "cuda"

    poems = [json.loads(l) for l in open(a.corpus, encoding="utf-8")]
    is_val = lambda p: int(hashlib.md5(p["poem_id"].encode()).hexdigest(), 16) % 1000 < a.val_frac * 1000
    train_p, val_p = [p for p in poems if not is_val(p)], [p for p in poems if is_val(p)]
    tok = Tokenizer.build(["\n".join(p["lines"]) for p in train_p], min_count=1, max_size=12000)
    train_x, val_x = examples(train_p, tok), examples(val_p, tok)
    val_pairs = [e for e in val_x if e[1] > 0]
    # the same replies after a call from another held-out pair (fixed derangement) → how much the call is used
    rot = val_pairs[1:] + val_pairs[:1]
    val_shuffled = [(c[:k] + r[k_r:], k) for (c, k), (r, k_r) in zip(rot, val_pairs)]
    prompt = "\n".join(normalize(l) for l in a.prompt.split("\n"))

    graph = load_graph(os.path.join(GRAPHS, f"{a.sex}_{a.region}.pt"), wiring=a.wiring, seed=a.seed)
    mb = load_mb(os.path.join(GRAPHS, f"{a.sex}_mb.pt"), a.region, graph["n"], wiring=a.wiring, seed=a.seed) if a.mb else None
    model = FlyBrainLM(graph, len(tok), d_model=a.d_model, ticks=a.ticks, alpha=a.alpha, learn_edges=not a.no_learn_edges,
                       readout_hidden=a.readout_hidden, learn_tau=not a.no_learn_tau, dropout=a.dropout,
                       interface="neuron", sep_id=SEP, mb=mb, mb_decay=a.mb_decay).to(dev)
    masks = verse_masks(tok)

    label = f"{POETS[a.sex]['name']} · books" + (" · random wiring" if a.wiring == "random" else "") + (" · mushroom-body memory" if a.mb else "")
    params = model.n_params()
    log = RunLogger(a.run_id, {
        "label": label, "poet": POETS[a.sex]["name"], "sex": a.sex, "connectome": POETS[a.sex]["connectome"], "args": vars(a),
        "graph": {"neurons": graph["n"], "edges": int(len(graph["pre"])), "synapses": int(graph["syn"].sum()), "interface": "neuron",
                  "input_neurons": int(len(model.in_idx)), "audio_neurons": int(len(model.au_idx)), "output_channels": model.n_out,
                  "output_neurons": int(len(model.out_idx)), "min_syn": graph.get("min_syn"), "wiring": a.wiring,
                  "mb_plastic_edges": int(len(model.mb_kc)) if a.mb else 0, "mb_mbons": model.n_mbon if a.mb else 0,
                  "mb_dan_edges": int(len(model.mb_dan)) if a.mb else 0},
        "params": {"total": sum(params.values()), "edges": params.get("edge_logscale", 0), "by_tensor": params},
        "data": {"corpus": os.path.relpath(a.corpus, ROOT), "poems_train": len(train_p), "poems_val": len(val_p),
                 "examples_train": len(train_x), "examples_val": len(val_x), "val_pairs": len(val_pairs), "vocab": len(tok),
                 "tokens_train": sum(len(s) for s, _ in train_x)},
        "vocab": len(tok), "prompt": prompt,
    }, sample_every_s=0)
    a.data = os.path.join(log.dir, "data")  # where load_model finds the tokenizer
    os.makedirs(a.data, exist_ok=True)
    tok.save(os.path.join(a.data, "tokenizer.json"))
    epochs_txt = open(os.path.join(log.dir, "epochs.txt"), "a", encoding="utf-8")

    edge = [model.edge_logscale] if model.edge_logscale is not None else []
    mbp = [p for n, p in model.named_parameters() if n.startswith("mb_")]
    other = [p for n, p in model.named_parameters() if n != "edge_logscale" and not n.startswith("mb_")]
    opt = torch.optim.AdamW([{"params": other, "lr": a.lr, "weight_decay": a.weight_decay},
                             {"params": edge, "lr": a.edge_lr, "weight_decay": 0.0},
                             {"params": mbp, "lr": a.lr, "weight_decay": 0.0}])
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = math.ceil(len(train_x) / a.batch)
    total = steps_per_epoch * a.epochs

    def loss_on(exs, reply_only=False):
        tot, n = 0.0, 0
        for i in range(0, len(exs), a.batch):
            inp, tgt, mask = batch(exs[i:i + a.batch], reply_only)
            logits, _ = model(inp.to(dev), grad_ckpt=False)
            m = int(mask.sum())
            tot += masked_ce(logits, tgt.to(dev), mask.to(dev)).item() * m
            n += m
        return tot / max(1, n)

    def evaluate():
        model.eval()
        with torch.no_grad():
            out = {"val_loss": loss_on(val_x), "val_loss_reply": loss_on(val_pairs, True)}
            out["val_loss_reply_wrong_call"] = loss_on(val_shuffled, True)
            out["context_gain"] = out["val_loss_reply_wrong_call"] - out["val_loss_reply"]
            if a.mb:
                model.mb_off = True
                out["val_loss_reply_no_memory"] = loss_on(val_pairs, True)
                model.mb_off = False
                out["memory_gain"] = out["val_loss_reply_no_memory"] - out["val_loss_reply"]
                out["mb_decay_mean"] = float(torch.sigmoid(model.mb_decay_logit).mean())
                out["mb_eta_mean"] = float(model.mb_eta.mean())
        out["val_ppl"] = math.exp(min(out["val_loss"], 20))
        model.train()
        return out

    def answer(epoch, res):
        model.eval()
        r = gen_reply(model, tok, prompt, mode="verse", n=16, masks=masks, style="cont")
        model.train()
        best = r["candidates"][:a.show]
        for c in best:
            log.sample(step, f"epoch {epoch}", prompt=prompt, reply=c["reply"], mode="verse (best of 16)", logp=c["logp"], **c["scores"])
        head = (f"=== EPOCH {epoch} · step {step} · {log.elapsed() / 60:.0f} min · val_loss {res['val_loss']:.3f} "
                f"(ppl {res['val_ppl']:.0f}) · reply loss {res['val_loss_reply']:.3f} · context gain {res['context_gain']:+.3f}"
                + (f" · memory gain {res['memory_gain']:+.3f} · λ {res['mb_decay_mean']:.3f} η {res['mb_eta_mean']:+.2f}" if a.mb else ""))
        body = "\n".join(f"[{i + 1}] " + c["reply"].replace("\n", " / ")
                         + f"   (meter {'✓' if c['scores']['meter_ok'] else '✗'}, rhyme {'✓' if c['scores']['rhyme_ok'] else '✗'})"
                         for i, c in enumerate(best))
        epochs_txt.write(head + "\n" + body + "\n=== END\n")
        epochs_txt.flush()
        print(head + "\n" + body, flush=True)

    def save(tag):
        torch.save({"model": model.state_dict(), "step": step, "epoch": epoch, "args": vars(a)},
                   os.path.join(log.dir, "checkpoints", f"{tag}.pt"))

    step, epoch, best_loss, bad = 0, 0, float("inf"), 0
    state, note = "finished", None
    model.train()
    try:
        with log.sampling():
            res = evaluate()
            log.eval(step, "epoch 0", epoch=0, **res)
            answer(0, res)
        for epoch in range(1, a.epochs + 1):
            order = list(range(len(train_x)))
            random.shuffle(order)
            t_last, tok_last = time.time(), 0
            for i in range(0, len(order), a.batch):
                inp, tgt, mask = batch([train_x[j] for j in order[i:i + a.batch]])
                scale = min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))
                for g, lr in zip(opt.param_groups, base_lrs):
                    g["lr"] = lr * scale
                logits, _ = model(inp.to(dev))
                loss = masked_ce(logits, tgt.to(dev), mask.to(dev))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                if not torch.isfinite(gn):
                    log.metrics(step, f"epoch {epoch}", event="nonfinite_grad")
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                step += 1
                tok_last += int(mask.sum())
                if step % a.log_every == 0:
                    torch.cuda.synchronize()
                    dt = time.time() - t_last
                    log.metrics(step, f"epoch {epoch}", epoch=epoch, train_loss=loss.item(), lr=opt.param_groups[0]["lr"],
                                grad_norm=float(gn), tokens_per_s=tok_last / max(dt, 1e-6),
                                gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9, rho=model.rho.item(), in_gain=model.in_gain.item(),
                                tau_mean=float(model.rates().mean()) if model.alpha_logit is not None else None,
                                mb_decay_mean=float(torch.sigmoid(model.mb_decay_logit).mean()) if a.mb else None,
                                mb_eta_mean=float(model.mb_eta.mean()) if a.mb else None)
                    t_last, tok_last = time.time(), 0
            with log.sampling():
                res = evaluate()
                log.eval(step, f"epoch {epoch}", epoch=epoch, **res)
                save("latest")
                if res["val_loss"] < best_loss:
                    best_loss, bad = res["val_loss"], 0
                    save("best")
                else:
                    bad += 1
                answer(epoch, res)
            if bad >= a.patience:
                note = f"early stop: {bad} epochs without a better val_loss (best {best_loss:.3f})"
                break
            if a.max_hours and log.elapsed() > a.max_hours * 3600:
                note = f"time limit {a.max_hours} h"
                break
    except KeyboardInterrupt:
        state, note = "stopped", "interrupted"
    finally:
        save("end")
        epochs_txt.write(f"=== DONE {state} {note or ''}\n")
        epochs_txt.flush()
        log.close(state, note)


if __name__ == "__main__":
    main()
