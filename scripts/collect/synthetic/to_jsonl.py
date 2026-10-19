"""Convert the synthetic duel parts (data/raw/synthetic_duels/part_*.jsonl) into extracted.jsonl records for
scripts/build_dataset.py: one record per turn, in order, each replying to the previous turn of its duel.
Every record is marked synthetic so it stays distinguishable from real collected verse."""
import glob
import json
import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "raw", "synthetic_duels")
out, duels = [], 0
for path in sorted(glob.glob(os.path.join(D, "part_*.jsonl"))):
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        duels += 1
        prev = None
        for k, t in enumerate(d.get("turns", [])):
            rid = f"{d['duel_id']}:{k}"
            out.append({"text": t["text"], "kind": "couplet", "id": rid, "reply_to": prev, "thread_id": d["duel_id"],
                        "position": k, "author": t.get("poet"), "duel": True, "title": f"{d.get('theme', '')} ({d.get('tone', '')})",
                        "category": f"synthetic duel · {d.get('tone', '')}", "category_norm": "tsiattista", "dialect": "cypriot",
                        "synthetic": True, "tone": d.get("tone"), "theme": d.get("theme"), "url": None})
            prev = rid
with open(os.path.join(D, "extracted.jsonl"), "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{duels} duels, {len(out)} turns -> {os.path.join(D, 'extracted.jsonl')}")
