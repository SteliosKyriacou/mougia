"""Does the reply depend on the call? Two diagnostics for the paper.

1. Data: content-word overlap between call and reply in real pairs vs. randomly re-paired calls.
2. Model: reply loss given the true call vs. a mismatched call vs. no call. If a model uses the context,
   the true call must give clearly lower loss on the reply.

  python3 -I mougia_train/diagnose_context.py runs/<run>/checkpoints/<ckpt>.pt [more checkpoints...]
"""
import json
import os
import random
import re
import sys
import unicodedata

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from mougia_train.sample_checkpoint import load_model  # noqa: E402
from mougia_train.tokenizer import BOS, SEP  # noqa: E402
from mougia_train.train import DATA  # noqa: E402

STOP = set("και τζαι τζιαι τζι τζ να το τα τη την της του των τον τους τες τις σε στο στη στην στον μου σου μας σας εν ένι που με για ο η οι ένα μια μα αν δεν μεν θα ως όπως ότι πως είναι ήταν έν εις εγιώ εσού ".split())


def words(text):
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return {w[:5] for w in re.findall(r"[α-ως]+", t) if len(w) > 2 and w not in STOP}


def overlap(a, b):
    A, B = words(a), words(b)
    return len(A & B) / max(1, len(A | B))


def data_check(tok):
    out = {}
    for split in ("train", "val"):
        pairs = torch.load(os.path.join(DATA, f"pairs_{split}.pt"))
        for style in ("duel", "cont"):
            ps = [p for p in pairs if p["style"] == style]
            calls = [tok.decode(p["call"]) for p in ps]
            reps = [tok.decode(p["reply"]) for p in ps]
            real = sum(overlap(c, r) for c, r in zip(calls, reps)) / len(ps)
            rng = random.Random(0)
            shuf = calls[:]
            rng.shuffle(shuf)
            fake = sum(overlap(c, r) for c, r in zip(shuf, reps)) / len(ps)
            out[f"{split}_{style}"] = {"n": len(ps), "overlap_real": round(real, 4), "overlap_shuffled": round(fake, 4)}
    return out


@torch.no_grad()
def model_check(model, pairs, dev="cuda"):
    rng = random.Random(0)
    others = [p["call"] for p in pairs]
    rng.shuffle(others)
    res = {}
    for cond in ("true_call", "other_call", "no_call"):
        tot, n = 0.0, 0
        for i, p in enumerate(pairs):
            if cond == "true_call":
                call = p["call"]
            elif cond == "other_call":
                call = others[i] if others[i] is not p["call"] else others[(i + 1) % len(others)]
            else:
                call = [BOS] + p["call"][-2:]  # <bos> <sep> <style>
            seq = torch.tensor(call + p["reply"], device=dev)[None]
            logits, _ = model(seq[:, :-1], grad_ckpt=False)
            k = len(call)
            loss = F.cross_entropy(logits[0, k - 1:], seq[0, k:], reduction="sum")
            tot += loss.item()
            n += seq.shape[1] - k
        res[cond] = round(tot / n, 4)
    res["context_gain_vs_other"] = round(res["other_call"] - res["true_call"], 4)
    return res


def main():
    ckpts = sys.argv[1:]
    report = {}
    for ck in ckpts:
        model, tok, meta = load_model(ck)
        if "data" not in report:
            report["data"] = data_check(tok)
            print("DATA", json.dumps(report["data"], indent=1))
        val = torch.load(os.path.join(DATA, "pairs_val.pt"))
        r = {style: model_check(model, [p for p in val if p["style"] == style]) for style in ("duel", "cont")}
        report[ck] = r
        print(ck, json.dumps(r))
        del model
        torch.cuda.empty_cache()
    out = os.path.join(HERE, "..", "runs", "_diagnostics_context.json")
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
