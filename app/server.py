"""Mougia dataset browser: a small stdlib HTTP server over data/processed/{records,couplets,pairs}.jsonl.

Run:  python3 app/server.py [--port 8765]   then open http://127.0.0.1:8765
"""
import argparse
import collections
import http.server
import json
import os
import re
import unicodedata
import urllib.parse

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "data", "processed")
STATIC = os.path.join(ROOT, "static")

GROUPS = (("web_", "Web"), ("book_", "Books"), ("corpus_", "Corpora"), ("youtube", "YouTube"))
SOURCE_NAMES = {
    "book_sakellarios1891": "Sakellarios, Τα Κυπριακά (1891)",
    "web_poetryfromcyprus": "Κυπρίων Ποίηση blog",
    "web_shapxar": "Χάρης Σιαπ. — written duels",
    "web_cyprushunting": "Cyprus Hunting forum",
    "web_stixoi": "stixoi.info — Cypriot songs",
    "web_tsiatista_wordpress": "tsiatista.wordpress",
    "web_kupriakatsiatismata": "Κυπριακά τσιατίσματα blog",
    "corpus_tsiattista_muse": "Tsiattista Muse (gold)",
    "corpus_mined": "Mined from GRDD corpora",
    "youtube": "YouTube duels",
}


def fold(s):
    """Accent- and case-insensitive search key."""
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch)).replace("ς", "σ")


def group_of(src):
    return next((g for p, g in GROUPS if src.startswith(p)), "Other")


def load():
    records = []
    with open(os.path.join(DATA, "records.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            r["i"] = i
            r["_fold"] = fold(r["text"] + " " + (r.get("author") or ""))
            records.append(r)
    threads = collections.defaultdict(list)
    for r in records:
        threads[(r["src"], r["thread"])].append(r)
    meta = {}
    for key, rs in threads.items():
        rs.sort(key=lambda r: (r["order"] if r["order"] is not None else 1e9, r.get("segment") or 0, r["n"]))
        authors = {r["author"] for r in rs if r.get("author") and r["status"] != "whole_poem"}
        kept = sum(r["status"] == "kept" for r in rs)
        battle = kept >= 2 and (len(authors) >= 2 or any(r["duel"] for r in rs) or key[0] == "youtube")
        title = next((r["title"] for r in rs if r.get("title")), None) or key[1]
        meta[key] = {"src": key[0], "thread": key[1], "title": str(title), "n": len(rs), "kept": kept,
                     "authors": sorted(authors)[:12], "n_authors": len(authors), "battle": battle,
                     "date": next((r["date"] for r in rs if r.get("date")), None), "url": next((r["url"] for r in rs if r.get("url")), None)}
    pairs = [json.loads(l) for l in open(os.path.join(DATA, "pairs.jsonl"), encoding="utf-8")]
    couplets = {}
    with open(os.path.join(DATA, "couplets.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            couplets[c["id"]] = {"lines": c["lines"], "syl": c["syllables"], "meter_ok": c["meter_ok"], "rhyme_ok": c["rhyme_ok"]}
    return records, threads, meta, pairs, couplets


RECORDS, THREADS, META, PAIRS, COUPLETS = load()
PUBLIC = ("src", "n", "thread", "order", "title", "author", "date", "url", "text", "kind", "category", "rid", "reply_to",
          "reply_to_user", "duel", "start", "status", "cid", "cy", "meter_ok", "rhyme_ok", "syl", "i")


def public(r):
    out = {k: r.get(k) for k in PUBLIC}
    c = COUPLETS.get(r.get("cid"))
    if c:  # kept or duplicate-of: show the normalized 15-syllable lines
        out.update(lines=c["lines"], syl=c["syl"], meter_ok=c["meter_ok"], rhyme_ok=c["rhyme_ok"])
    return out


def overview():
    by_src = collections.defaultdict(lambda: collections.Counter())
    for r in RECORDS:
        by_src[r["src"]][r["status"]] += 1
    threads_by_src = collections.Counter(m["src"] for m in META.values())
    battles_by_src = collections.Counter(m["src"] for m in META.values() if m["battle"])
    sources = []
    for src, cnt in by_src.items():
        sources.append({"src": src, "name": SOURCE_NAMES.get(src, src.split("_", 1)[-1].replace("_", " ")), "group": group_of(src),
                        "kept": cnt["kept"], "records": sum(cnt.values()), "threads": threads_by_src[src], "battles": battles_by_src[src]})
    sources.sort(key=lambda s: (-s["kept"], s["src"]))
    status = collections.Counter(r["status"] for r in RECORDS)
    kept = [r for r in RECORDS if r["status"] == "kept"]
    return {"sources": sources, "status": status, "kept": len(kept),
            "good": sum(bool(r["meter_ok"] and r["rhyme_ok"]) for r in kept),
            "pairs": len(PAIRS), "good_pairs": sum(p["good"] for p in PAIRS),
            "battles": sum(m["battle"] for m in META.values()), "threads": len(META)}


def list_threads(q):
    src, only_battles = q.get("src"), q.get("battles") == "1"
    term = fold(q.get("q", ""))
    out = [m for m in META.values() if (not src or m["src"] == src) and (not only_battles or m["battle"])
           and (q.get("all") == "1" or m["kept"] > 0)]
    if term:
        out = [m for m in out if term in fold(m["title"]) or any(term in r["_fold"] for r in THREADS[(m["src"], m["thread"])])]
    out.sort(key=lambda m: (-m["battle"], -(m["kept"]), m["title"]))
    return page(out, q)


def get_thread(q):
    key = (q.get("src"), q.get("thread"))
    if key not in THREADS:
        return {"error": "thread not found"}
    return {"meta": META[key], "records": [public(r) for r in THREADS[key]]}


def search(q):
    term = fold(q.get("q", "")).strip()
    kept_only = q.get("all") != "1"
    out = [r for r in RECORDS if (not kept_only or r["status"] == "kept") and (not term or term in r["_fold"])
           and (not q.get("src") or r["src"] == q["src"]) and (q.get("good") != "1" or (r["meter_ok"] and r["rhyme_ok"]))]
    return page([public(r) for r in out], q)


def list_pairs(q):
    out = [p for p in PAIRS if (q.get("good") != "1" or p["good"]) and (not q.get("src") or p["source"] == q["src"])]
    return page(out, q)


def page(items, q, size=50):
    p = max(0, int(q.get("page", 0) or 0))
    return {"total": len(items), "page": p, "size": size, "items": items[p * size:(p + 1) * size]}


# ---------- Training runs (written by mougia_train/runlog.py) ----------
RUNS = os.environ.get("MOUGIA_RUNS") or os.path.join(ROOT, "..", "runs")
_cache = {}


def read_jsonl(path):
    """Read a JSONL file, cached by (mtime, size); tolerates a partially written last line."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return []
    key = (st.st_mtime_ns, st.st_size)
    hit = _cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    _cache[path] = (key, rows)
    return rows


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def run_dirs():
    if not os.path.isdir(RUNS):
        return []
    out = []
    for name in os.listdir(RUNS):
        cfg = read_json(os.path.join(RUNS, name, "config.json"))
        if cfg:
            out.append((cfg.get("started_at") or "", name, cfg))
    return sorted(out)


def list_runs(q):
    runs = []
    for slot, (_, name, cfg) in enumerate(run_dirs()):
        d = os.path.join(RUNS, name)
        metrics, evals = read_jsonl(os.path.join(d, "metrics.jsonl")), read_jsonl(os.path.join(d, "eval.jsonl"))
        runs.append({"run": name, "slot": slot, "label": cfg.get("label") or name, "config": cfg,
                     "status": read_json(os.path.join(d, "status.json")), "last_metrics": metrics[-1] if metrics else None,
                     "last_eval": evals[-1] if evals else None, "n_metrics": len(metrics), "n_eval": len(evals),
                     "n_samples": len(read_jsonl(os.path.join(d, "samples.jsonl")))})
    return {"runs_dir": os.path.abspath(RUNS), "runs": runs}


def downsample(rows, n=1500):
    if len(rows) <= n:
        return rows
    stride = len(rows) / n
    return [rows[int(i * stride)] for i in range(n)] + [rows[-1]]


def run_series(q):
    d = os.path.join(RUNS, os.path.basename(q.get("run", "")))
    return {"metrics": downsample(read_jsonl(os.path.join(d, "metrics.jsonl"))), "eval": read_jsonl(os.path.join(d, "eval.jsonl"))}


def run_samples(q):
    names = [os.path.basename(q["run"])] if q.get("run") else [n for _, n, _ in run_dirs()]
    rows = [{**r, "run": n} for n in names for r in read_jsonl(os.path.join(RUNS, n, "samples.jsonl"))]
    rows.sort(key=lambda r: (r.get("time") or "", r.get("step") or 0), reverse=True)
    return page(rows, q, size=int(q.get("size", 20) or 20))


ROUTES = {"/api/runs": list_runs, "/api/run_series": run_series, "/api/run_samples": run_samples,
          "/api/overview": lambda q: overview(), "/api/threads": list_threads, "/api/thread": get_thread,
          "/api/search": search, "/api/pairs": list_pairs}
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".svg": "image/svg+xml"}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path in ROUTES:
            q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
            body = json.dumps(ROUTES[url.path](q), ensure_ascii=False).encode()
            return self.send(200, "application/json; charset=utf-8", body)
        name = "index.html" if url.path in ("/", "") else url.path.lstrip("/")
        path = os.path.normpath(os.path.join(STATIC, name))
        if not path.startswith(STATIC) or not os.path.isfile(path):
            path = os.path.join(STATIC, "index.html")  # client-side routes
        with open(path, "rb") as f:
            self.send(200, TYPES.get(os.path.splitext(path)[1], "application/octet-stream"), f.read())

    def send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"Mougia browser on http://{a.host}:{a.port}  ({len(RECORDS)} records, {len(META)} threads)")
    http.server.ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
