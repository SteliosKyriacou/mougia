"""Generate reply samples from a saved checkpoint and append them to that run's samples.jsonl.

  python3 -I mougia_train/sample_checkpoint.py runs/<run_id>/checkpoints/latest.pt [--n 1]

Samples are tagged with the checkpoint's step and the run's training time at that step (from metrics.jsonl), so they
line up with the 15-minute samples in the Training tab.
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from mougia_train.model import FlyBrainLM, load_graph  # noqa: E402
from mougia_train.generate import reply as gen_reply  # noqa: E402
from mougia_train.tokenizer import Tokenizer  # noqa: E402
from mougia_train.train import DATA, GRAPHS  # noqa: E402


def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    tok = Tokenizer.load(os.path.join(DATA, "tokenizer.json"))
    graph = load_graph(os.path.join(GRAPHS, f"{a['sex']}_{a['region']}.pt"), wiring=a["wiring"], max_edges=a.get("max_edges") or None, seed=a["seed"])
    model = FlyBrainLM(graph, len(tok), d_model=a["d_model"], ticks=a["ticks"], alpha=a["alpha"],
                       learn_edges=not a["no_learn_edges"], readout_hidden=a["readout_hidden"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), tok, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--n", type=int, default=1, help="replies per prompt")
    ap.add_argument("--temperature", type=float, default=0.8)
    a = ap.parse_args()
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(a.checkpoint)))
    model, tok, ck = load_model(a.checkpoint)
    step = ck.get("step", 0)
    elapsed, phase = None, None
    mpath = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(mpath):
        for line in open(mpath, encoding="utf-8"):
            r = json.loads(line)
            if r.get("step", 0) <= step:
                elapsed, phase = r.get("elapsed_s"), r.get("phase")
    prompts = json.load(open(os.path.join(DATA, "sample_prompts.json"), encoding="utf-8"))
    with open(os.path.join(run_dir, "samples.jsonl"), "a", encoding="utf-8") as f:
        for p in prompts:
            for _ in range(a.n):
                v = gen_reply(model, tok, p["text"], mode="verse", n=16, temperature=a.temperature)
                reply = v["reply"]
                rec = {"step": step, "elapsed_s": elapsed, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "phase": phase,
                       "prompt": p["text"], "reply": reply, "prompt_id": p["id"], "source": f"checkpoint {os.path.basename(a.checkpoint)}",
                       "mode": "verse (best of 16)", **v["scores"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print("—", reply.replace("\n", " / "))


if __name__ == "__main__":
    main()
