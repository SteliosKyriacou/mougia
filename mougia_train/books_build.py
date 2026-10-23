"""Books-only dataset, step 2: merge poem annotations into training data for the two-stage Krikri fine-tune.

Inputs (data/processed/books_v1/): poems.jsonl (books_prepare.py), extra_poems.jsonl (verse extracted from the mixed
books), annot/*.jsonl (per-poem dialect + speaker turns, read and annotated by hand).
Outputs:
  stage1_{train,val}.jsonl  {"poem_id", "source", "title", "text"} — every Cypriot-dialect poem (and distich), one doc each
  stage2_{train,val}.jsonl  {"poem_id", "level": "couplet|turn", "call", "reply", "call_speaker", "reply_speaker"}
      reconstructed battles: consecutive speech turns by different speakers. "turn" = whole speeches (≤ 16 lines each),
      "couplet" = the last couplet of the call and the first couplet of the reply (the τσιαττιστό-sized exchange).
  stats.json
Splits are by poem (≈10%), so no poem is in both train and val.
"""
import collections
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from mougia_train.books_prepare import normalize  # noqa: E402

D = os.path.join(HERE, "..", "data", "processed", "books_v1")
VAL_FRAC = 0.10
MAX_TURN_LINES = 16
# Non-verse lines found while annotating (Wikisource maintenance notices, newspaper masthead, editorial prose,
# a stanza duplicated from another poem): left out of the stage-1 text.
EXCLUDE_LINES = {"wiki-η-9η-ιουλίου-του-1821": [(0, 62)], "wiki-τα-προξενεία-της-αφροδίτης": [(0, 11), (48, 56)]}
# Chronicles with only one or two real exchanges (annotator's advice): kept for stage 1, not used as battles.
NOT_BATTLES = {"sak-song-25", "sak-song-32-b", "sak-song-33"}


def is_val(pid):
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 1000 < VAL_FRAC * 1000


def main():
    poems = {p["poem_id"]: p for p in map(json.loads, open(os.path.join(D, "poems.jsonl"), encoding="utf-8"))}
    extra = os.path.join(D, "extra_poems.jsonl")
    if os.path.exists(extra):
        for p in map(json.loads, open(extra, encoding="utf-8")):
            p["original"] = p["lines"]
            p["lines"] = [normalize(l) for l in p["lines"] if l.strip()]
            poems[p["poem_id"]] = p
    annot = {}
    for path in sorted(glob.glob(os.path.join(D, "annot", "*.jsonl"))):
        for r in map(json.loads, open(path, encoding="utf-8")):
            annot[r["poem_id"]] = r

    stage1 = {"train": [], "val": []}
    stage2 = {"train": [], "val": []}
    stats = collections.Counter()
    for pid, p in poems.items():
        a = annot.get(pid)
        if a is None and p["source"] != "sakellarios1891-distichs":
            stats["not_annotated"] += 1
            continue
        dialect = a["dialect"] if a else "cypriot"  # Sakellarios distichs are Cypriot folk verse by provenance
        stats[f"dialect_{dialect}"] += 1
        if dialect != "cypriot":
            continue  # Cypriot only (Standard-Greek and mixed poems are left out)
        split = "val" if is_val(pid) else "train"
        drop = {i for lo, hi in EXCLUDE_LINES.get(pid, []) for i in range(lo, hi + 1)}
        keep = [l for i, l in enumerate(p["lines"]) if i not in drop]
        stage1[split].append({"poem_id": pid, "source": p["source"], "title": p.get("title"), "text": "\n".join(keep)})
        stats[f"stage1_{split}_lines"] += len(keep)
        if not (a and a.get("is_dialogue")) or pid in NOT_BATTLES:
            continue
        stats["dialogue_poems"] += 1
        speeches = [t for t in a["turns"] if t["kind"] == "speech"]
        stats["speech_turns"] += len(speeches)
        for prev, cur in zip(speeches, speeches[1:]):
            if prev["speaker"] == cur["speaker"]:
                continue
            call = p["lines"][prev["start"]:prev["end"] + 1]
            reply = p["lines"][cur["start"]:cur["end"] + 1]
            base = {"poem_id": pid, "call_speaker": prev["speaker"], "reply_speaker": cur["speaker"], "title": p.get("title")}
            stage2[split].append({**base, "level": "turn", "call": "\n".join(call[-MAX_TURN_LINES:]), "reply": "\n".join(reply[:MAX_TURN_LINES])})
            if len(call) >= 2 and len(reply) >= 2:
                stage2[split].append({**base, "level": "couplet", "call": "\n".join(call[-2:]), "reply": "\n".join(reply[:2])})
            stats[f"stage2_{split}_exchanges"] += 1

    for name, parts in (("stage1", stage1), ("stage2", stage2)):
        for split, rows in parts.items():
            with open(os.path.join(D, f"{name}_{split}.jsonl"), "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats[f"{name}_{split}_records"] = len(rows)
    json.dump(dict(stats), open(os.path.join(D, "stats.json"), "w"), indent=2)
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main()
