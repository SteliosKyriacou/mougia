"""Replace Standard-Greek forms with their Cypriot equivalents in the synthetic duels (same syllable count),
in place. Couplets still failing scripts/check_couplets.py afterwards are listed in needs_rewrite.jsonl."""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from check_couplets import check  # noqa: E402

D = os.path.join(HERE, "..", "..", "..", "data", "raw", "synthetic_duels")
AUTO = {"πιο": "πκιο", "ήταν": "ήτουν", "εσένα": "εσέναν", "εμένα": "εμέναν", "έχω": "έσιω", "έχεις": "έσιεις",
        "έχει": "έσιει", "έχουμε": "έσιουμεν", "έχουμεν": "έσιουμεν", "έχετε": "έσιετε", "έχουν": "έσιουν",
        "τίποτα": "τίποτε", "σήμερα": "σήμερον"}
WORD = re.compile(r"[^\W\d_]+")


def fix(text):
    def sub(m):
        w = m.group(0)
        rep = AUTO.get(w.lower())
        if not rep:
            return w
        return rep[0].upper() + rep[1:] if w[0].isupper() else rep
    return WORD.sub(sub, text)


changed, todo = 0, []
for path in sorted(glob.glob(os.path.join(D, "part_*.jsonl"))):
    duels = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for d in duels:
        for k, t in enumerate(d["turns"]):
            new = fix(t["text"])
            if new != t["text"]:
                t["text"] = new
                changed += 1
            ok, probs, _ = check(t["text"])
            if not ok:
                prev = d["turns"][k - 1]["text"] if k else None
                todo.append({"part": os.path.basename(path), "duel_id": d["duel_id"], "turn": k, "poet": t["poet"],
                             "previous_turn": prev, "text": t["text"], "problems": probs})
    with open(path, "w", encoding="utf-8") as f:
        for d in duels:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
with open(os.path.join(D, "needs_rewrite.jsonl"), "w", encoding="utf-8") as f:
    for r in todo:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"auto-fixed {changed} couplets; {len(todo)} still need a rewrite -> needs_rewrite.jsonl")
