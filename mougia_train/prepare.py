"""Build tokenized training data for the fly-brain poets.

Writes data/processed/train/:
  tokenizer.json   syllable vocabulary
  lm_train.pt / lm_val.pt          flat token streams of Cypriot dialect text (pretraining)
  couplets_train.pt / couplets_val.pt   lists of token-id lists "<bos> line1 <nl> line2 <eos>"
  pairs_train.pt / pairs_val.pt    lists of (call_ids, reply_ids)
  sample_prompts.json              fixed held-out calls answered every 15 min of training
  stats.json

Splits are by thread/source document so a duel never leaks between train and val. Any pretraining line that
matches a held-out couplet line is dropped from the LM stream.
"""
import glob
import hashlib
import json
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
from greek_meter import dedupe_key  # noqa: E402
from mougia_train.tokenizer import BOS, EOS, NL, SEP, Tokenizer  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(PROC, "train")
LM_SOURCES = [
    "corpus_grdd_plus/repo/data/final_cypriot.txt",
    "corpus_grdd_zenodo/extracted/*/Cypriot/Cypriot_raw/*.txt",
    "corpus_grdd_zenodo/extracted/*/SMS_CG/cg_*.txt",
    "book_sakellarios1891/songs_text.txt",
]
VAL_FRAC_COUPLETS, VAL_FRAC_PAIRS, VAL_FRAC_LM = 0.05, 0.15, 0.01
N_SAMPLE_PROMPTS = 4


def bucket(key, frac):
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 10_000 < frac * 10_000


def main():
    os.makedirs(OUT, exist_ok=True)
    couplets = [json.loads(l) for l in open(os.path.join(PROC, "couplets.jsonl"), encoding="utf-8")]
    pairs = [json.loads(l) for l in open(os.path.join(PROC, "pairs.jsonl"), encoding="utf-8")]
    records = {}
    for l in open(os.path.join(PROC, "records.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        if r.get("cid") and r["status"] == "kept":
            records[r["cid"]] = r["src"] + "|" + r["thread"]
    by_id = {c["id"]: c for c in couplets}

    # Pairs split by thread; couplets that appear in a val pair are val too.
    val_threads = {records.get(p["reply"], p["reply"]) for p in pairs if bucket(records.get(p["reply"], p["reply"]), VAL_FRAC_PAIRS)}
    pair_val = [p for p in pairs if records.get(p["reply"], p["reply"]) in val_threads]
    pair_train = [p for p in pairs if records.get(p["reply"], p["reply"]) not in val_threads]
    val_cids = {p[k] for p in pair_val for k in ("call", "reply")}
    coup_val = [c for c in couplets if c["id"] in val_cids or bucket(records.get(c["id"], c["id"]), VAL_FRAC_COUPLETS)]
    val_set = {c["id"] for c in coup_val}
    coup_train = [c for c in couplets if c["id"] not in val_set]
    held_lines = {dedupe_key(line) for c in coup_val for line in c["lines"]}

    # Pretraining text: line-deduplicated, held-out couplet lines removed.
    lm_lines, seen, dropped = [], set(), 0
    for pattern in LM_SOURCES:
        for path in sorted(glob.glob(os.path.join(RAW, pattern))):
            for line in open(path, encoding="utf-8", errors="ignore"):
                line = line.strip()
                k = dedupe_key(line)
                if len(k) < 8 or k in seen:
                    continue
                seen.add(k)
                if k in held_lines:
                    dropped += 1
                    continue
                lm_lines.append(line)
    lm_lines += ["\n".join(c["lines"]) for c in coup_train]  # couplets are also dialect text

    tok = Tokenizer.build(lm_lines + ["\n".join(c["lines"]) for c in couplets], min_count=3, max_size=12000)
    tok.save(os.path.join(OUT, "tokenizer.json"))

    random.seed(0)
    random.shuffle(lm_lines)
    n_val = int(len(lm_lines) * VAL_FRAC_LM)

    def stream(lines):
        ids = []
        for l in lines:
            ids += tok.encode(l, bos=True, eos=True)
        return torch.tensor(ids, dtype=torch.int32)

    lm_val, lm_train = stream(lm_lines[:n_val]), stream(lm_lines[n_val:])
    enc_c = lambda c: tok.encode("\n".join(c["lines"]), bos=True, eos=True)
    enc_p = lambda p: (tok.encode(by_id[p["call"]]["text"], bos=True) + [SEP], tok.encode(by_id[p["reply"]]["text"], eos=True))
    torch.save(lm_train, os.path.join(OUT, "lm_train.pt"))
    torch.save(lm_val, os.path.join(OUT, "lm_val.pt"))
    torch.save([enc_c(c) for c in coup_train], os.path.join(OUT, "couplets_train.pt"))
    torch.save([enc_c(c) for c in coup_val], os.path.join(OUT, "couplets_val.pt"))
    torch.save([enc_p(p) for p in pair_train], os.path.join(OUT, "pairs_train.pt"))
    torch.save([enc_p(p) for p in pair_val], os.path.join(OUT, "pairs_val.pt"))

    # Fixed prompts for the 15-minute samples: held-out calls with strict meter + rhyme, from different threads.
    prompts, used = [], set()
    prio = {"web_shapxar": 0, "corpus_tsiattista_muse": 1, "web_tsiatista_wordpress": 2}
    for p in sorted(pair_val, key=lambda p: (prio.get(by_id[p["call"]]["source"], 9), -by_id[p["call"]]["cy_score"], p["call"])):
        c = by_id[p["call"]]
        th = records.get(p["call"])
        if c["meter_ok"] and c["rhyme_ok"] and th not in used:
            prompts.append({"id": c["id"], "text": c["text"], "true_reply": by_id[p["reply"]]["text"]})
            used.add(th)
        if len(prompts) == N_SAMPLE_PROMPTS:
            break
    for c in coup_val:  # top up from held-out couplets if there are too few duel prompts
        if len(prompts) >= N_SAMPLE_PROMPTS:
            break
        if c["meter_ok"] and c["rhyme_ok"] and c["id"] not in {p["id"] for p in prompts}:
            prompts.append({"id": c["id"], "text": c["text"], "true_reply": None})
    json.dump(prompts, open(os.path.join(OUT, "sample_prompts.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    stats = {"vocab": len(tok), "lm_train_tokens": int(len(lm_train)), "lm_val_tokens": int(len(lm_val)), "lm_lines": len(lm_lines),
             "lm_lines_dropped_as_heldout": dropped, "couplets_train": len(coup_train), "couplets_val": len(coup_val),
             "pairs_train": len(pair_train), "pairs_val": len(pair_val), "sample_prompts": len(prompts),
             "unk_rate_lm_val": round(float((lm_val == 1).float().mean()), 5)}
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
