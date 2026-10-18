"""Run extract_couplets.py over every text file of a source dir and write <dir>/extracted.jsonl.
Keep rule (verse-block heuristic): run_len>=3, or run_len>=2 and dialect_score>=1, or dialect_score>=DIAL_ISOLATED.
Usage: python3 -I build_auto_source.py SRC_DIR --source "Name" [--url-prefix https://archive.org/details/] [--glob 'original/*.txt'] [--iso 4]"""
import sys, os, glob, json, subprocess, argparse
ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--source", required=True); ap.add_argument("--url-prefix", default="")
ap.add_argument("--url", default=""); ap.add_argument("--glob", default="original/*.txt"); ap.add_argument("--iso", type=int, default=4)
ap.add_argument("--min-run3", type=int, default=3)
a = ap.parse_args()
EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract_couplets.py")
recs, seen = [], set()
for f in sorted(glob.glob(os.path.join(a.dir, a.glob))):
    item = os.path.splitext(os.path.basename(f))[0]
    url = a.url or (a.url_prefix + item if a.url_prefix else "")
    p = subprocess.run([sys.executable, "-I", EX, f, "--source", a.source, "--url", url], capture_output=True, text=True)
    for line in p.stdout.splitlines():
        r = json.loads(line)
        rl, d = r.get("run_len", 1), r["dialect_score"]
        if rl >= a.min_run3 or (rl >= 2 and d >= 1) or d >= a.iso:
            key = r["text"]
            if key in seen: continue
            seen.add(key); r["item"] = item; recs.append(r)
with open(os.path.join(a.dir, "extracted.jsonl"), "w", encoding="utf-8") as o:
    for r in recs: o.write(json.dumps(r, ensure_ascii=False) + "\n")
print(a.dir, len(recs))
