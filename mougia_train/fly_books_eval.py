"""Re-evaluate fly_books.py checkpoints on the held-out poems: reply loss with the true call, with a call from another
poem (context_gain), and with the mushroom-body memory off (memory_gain). Appends to runs/<id>/reeval.jsonl.

Example:
  python3 -I mougia_train/fly_books_eval.py runs/fly-books-kyriakou-mb/checkpoints/best.pt
"""
import argparse
import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from mougia_train.fly_books import reply_pairs, batch  # noqa: E402
from mougia_train.sample_checkpoint import load_model  # noqa: E402
from mougia_train.train import masked_ce  # noqa: E402


@torch.no_grad()
def loss_on(model, exs, bs=64):
    tot, n = 0.0, 0
    for i in range(0, len(exs), bs):
        inp, tgt, mask = batch(exs[i:i + bs], True)
        logits, _ = model(inp.cuda(), grad_ckpt=False)
        m = int(mask.sum())
        tot += masked_ce(logits, tgt.cuda(), mask.cuda()).item() * m
        n += m
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    for path in ap.parse_args().checkpoints:
        model, tok, ck = load_model(path)
        a = ck["args"]
        poems = [json.loads(l) for l in open(a["corpus"], encoding="utf-8")]
        val_p = [p for p in poems if int(hashlib.md5(p["poem_id"].encode()).hexdigest(), 16) % 1000 < a["val_frac"] * 1000]
        true, wrong = reply_pairs(val_p, tok)
        res = {"checkpoint": os.path.basename(path), "epoch": ck.get("epoch"), "step": ck["step"],
               "val_loss_reply": loss_on(model, true), "val_loss_reply_wrong_call": loss_on(model, wrong)}
        res["context_gain"] = res["val_loss_reply_wrong_call"] - res["val_loss_reply"]
        if model.mb:
            model.mb_off = True
            res["val_loss_reply_no_memory"] = loss_on(model, true)
            res["val_loss_reply_wrong_call_no_memory"] = loss_on(model, wrong)
            model.mb_off = False
            res["memory_gain"] = res["val_loss_reply_no_memory"] - res["val_loss_reply"]
            res["context_gain_no_memory"] = res["val_loss_reply_wrong_call_no_memory"] - res["val_loss_reply_no_memory"]
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}), flush=True)
        with open(os.path.join(os.path.dirname(os.path.dirname(path)), "reeval.jsonl"), "a") as f:
            f.write(json.dumps(res) + "\n")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
