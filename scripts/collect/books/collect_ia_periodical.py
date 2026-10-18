"""Download archive.org djvu.txt OCR for a list of identifiers into OUT/original/<id>.txt (skip existing).
Usage: python3 -I collect_ia_periodical.py ids.txt OUTDIR"""
import sys, os, requests, time
from concurrent.futures import ThreadPoolExecutor
ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
out = os.path.join(sys.argv[2], "original"); os.makedirs(out, exist_ok=True)
def fetch(i):
    p = os.path.join(out, i + ".txt")
    if os.path.exists(p) and os.path.getsize(p) > 0: return i, "skip"
    for attempt in range(3):
        try:
            m = requests.get(f"https://archive.org/metadata/{i}", timeout=60).json()
            names = [f["name"] for f in m.get("files", []) if f["name"].endswith("_djvu.txt")]
            if not names: return i, "no-djvu"
            r = requests.get(f"https://archive.org/download/{i}/{names[0]}", timeout=120)
            r.raise_for_status()
            open(p, "w", encoding="utf-8").write(r.text); return i, len(r.text)
        except Exception as e:
            time.sleep(3); err = str(e)
    return i, "ERR " + err
with ThreadPoolExecutor(8) as ex:
    for i, st in ex.map(fetch, ids): print(i, st, flush=True)
