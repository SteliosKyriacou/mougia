"""Helper for annotating the books dataset.

  python3 -I mougia_train/books_show.py list [prefix]          # poem ids, titles, line counts
  python3 -I mougia_train/books_show.py show <poem_id>         # numbered normalised lines (+ original)
  python3 -I mougia_train/books_show.py check <annot.jsonl>    # validate an annotation file

Annotation record (one per poem, JSONL):
  {"poem_id": "...", "dialect": "cypriot|mixed|smg", "is_dialogue": true|false,
   "turns": [{"speaker": "...", "kind": "speech|narration", "start": 0, "end": 5}],   # inclusive line ranges
   "summary": "one line in English", "notes": "..."}
For dialogue poems the turns must cover every line in order, without gaps or overlaps.
"""
import json
import os
import sys

POEMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "books_v1", "poems.jsonl")


def load():
    return {p["poem_id"]: p for p in map(json.loads, open(POEMS, encoding="utf-8"))}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    poems = load()
    if cmd == "list":
        pre = sys.argv[2] if len(sys.argv) > 2 else ""
        for pid, p in poems.items():
            if pid.startswith(pre):
                print(f"{pid}\t{len(p['lines'])} lines\t{p['title']}")
    elif cmd == "show":
        p = poems[sys.argv[2]]
        print(f"# {p['poem_id']} — {p['title']} ({p['source']})")
        for i, (l, o) in enumerate(zip(p["lines"], p["original"])):
            print(f"{i:4d}  {l}" + (f"\n      [orig] {o}" if "--orig" in sys.argv else ""))
    elif cmd == "check":
        ok = bad = 0
        for n, line in enumerate(open(sys.argv[2], encoding="utf-8")):
            r = json.loads(line)
            p = poems.get(r.get("poem_id"))
            errs = []
            if not p:
                errs.append("unknown poem_id")
            elif r.get("is_dialogue"):
                pos = 0
                for t in r.get("turns", []):
                    if t["start"] != pos or t["end"] < t["start"]:
                        errs.append(f"turn {t} does not start at line {pos}")
                        break
                    if t.get("kind") not in ("speech", "narration"):
                        errs.append(f"bad kind in {t}")
                    pos = t["end"] + 1
                if not errs and pos != len(p["lines"]):
                    errs.append(f"turns end at {pos - 1}, poem has {len(p['lines'])} lines")
                if sum(t["kind"] == "speech" for t in r.get("turns", [])) < 2:
                    errs.append("a dialogue needs at least two speech turns")
            if r.get("dialect") not in ("cypriot", "mixed", "smg"):
                errs.append("dialect must be cypriot|mixed|smg")
            if errs:
                bad += 1
                print(f"line {n + 1} {r.get('poem_id')}: " + "; ".join(errs))
            else:
                ok += 1
        print(f"{ok} ok, {bad} with errors")


if __name__ == "__main__":
    main()
