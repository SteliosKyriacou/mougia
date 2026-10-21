"""Mougia dataset browser: a small stdlib HTTP server over data/processed/{records,couplets,pairs}.jsonl.

Run:  python3 app/server.py [--port 8765]   then open http://127.0.0.1:8765
"""
import argparse
import collections
import http.server
import json
import os
import re
import time
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
    except (OSError, json.JSONDecodeError):
        return None


def run_dirs():
    if not os.path.isdir(RUNS):
        return []
    out = []
    for name in os.listdir(RUNS):
        if not os.path.isdir(os.path.join(RUNS, name)):
            continue  # stray files (e.g. logs) are not runs
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


# ---------- Brain viewer: neurons + weighted edges from raw connectomes or training checkpoints ----------
GRAPHS = os.path.join(ROOT, "..", "data", "connectomes", "processed")
POET = {"female": "Kyriakou", "male": "Markos"}
SUPERCLASS_GROUPS = [  # (label, female names, male names) — display grouping shared by both sexes
    ("Optic lobe", {"optic"}, {"ol_intrinsic"}),
    ("Visual projection", {"visual_projection", "visual_centrifugal"}, {"visual_projection", "visual_centrifugal", "ol_sensory"}),
    ("Central brain", {"central"}, {"cb_intrinsic"}),
    ("Sensory", {"sensory", "sensory_ascending"}, {"cb_sensory", "sensory_ascending", "sensory_descending"}),
    ("Ascending", {"ascending"}, {"ascending_neuron", "efferent_ascending"}),
    ("Descending", {"descending"}, {"descending_neuron", "efferent_descending"}),
    ("Motor / endocrine", {"motor", "endocrine"}, {"cb_motor", "cb_endocrine", "cb_efferent"}),
]
_brain_cache = collections.OrderedDict()


def _cached(key, fn, size=4):
    if key in _brain_cache:
        _brain_cache.move_to_end(key)
        return _brain_cache[key]
    val = fn()
    _brain_cache[key] = val
    while len(_brain_cache) > size:
        _brain_cache.popitem(last=False)
    return val


def _torch():
    import torch  # only the brain viewer needs torch
    return torch


def load_graph_file(name):
    torch = _torch()
    return _cached(("graph", name), lambda: torch.load(os.path.join(GRAPHS, os.path.basename(name) + ".pt")))


def brain_models(q):
    models = []
    for name in ("female_full", "male_full", "female_central", "male_central"):
        if os.path.exists(os.path.join(GRAPHS, name + ".pt")):
            sex, region = name.split("_")
            models.append({"id": f"raw:{name}", "group": "Connectomes (before training)", "graph": name,
                           "label": f"{POET[sex]} — {'full brain' if region == 'full' else 'central brain'}"})
    for _, run, cfg in reversed(run_dirs()):
        args = cfg.get("args") or {}
        sex, region = args.get("sex"), args.get("region", "full")
        ck = os.path.join(RUNS, run, "checkpoints")
        if not sex or not os.path.isdir(ck):
            continue
        files = sorted(os.listdir(ck), key=lambda f: os.path.getmtime(os.path.join(ck, f)))
        for f in files:
            if f.endswith(".pt"):
                models.append({"id": f"ckpt:{run}/{f}", "group": cfg.get("label") or run, "graph": f"{sex}_{region}",
                               "label": f.removesuffix(".pt").replace("end-", "end of "),
                               "time": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(os.path.join(ck, f))))})
    return {"models": models}


def _edges_for(model_id):
    """(pre, post, w, scale, info) for a model id; w is the signed effective weight, scale = exp(learned log-scale)."""
    torch = _torch()

    def compute():
        if model_id.startswith("raw:"):
            g = load_graph_file(model_id[4:])
            pre, post, syn, sign = g["pre"], g["post"], g["syn"], g["sign"]
            tot = torch.zeros(g["n"]).index_add_(0, post, syn).clamp_min(1.0)
            w = sign[pre] * syn / tot[post]
            return pre, post, w, torch.ones_like(w), {"step": 0}
        run, f = model_id[5:].split("/", 1)
        path = os.path.join(RUNS, os.path.basename(run), "checkpoints", os.path.basename(f))
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model"]
        theta = sd.get("edge_logscale")
        scale = torch.exp(theta.clamp(-6, 6)) if theta is not None else torch.ones_like(sd["w0"])
        rho = float(sd.get("rho", torch.tensor(1.0)))
        return sd["col"], sd["row"], sd["w0"] * scale * rho, scale, {"step": ck.get("step"), "rho": rho}
    return _cached(("edges", model_id), compute)


def brain_neurons(q):
    """Binary: Float32 xyz (normalised, y up) for every neuron, then Uint8 display group, then Uint8 role
    (1 = smell input, 2 = output, 3 = hearing input)."""
    import numpy as np
    g = load_graph_file(q.get("graph", "female_full"))
    pos = g["pos"].numpy().astype(np.float64)
    pos = pos - np.median(pos, 0)
    pos /= np.percentile(np.abs(pos), 99) or 1.0
    pos[:, 1] *= -1  # image y grows ventrally
    pos[:, 2] *= -1
    female = q.get("graph", "").startswith("female")
    grp = np.zeros(g["n"], np.uint8)
    for k, (_, fs, ms) in enumerate(SUPERCLASS_GROUPS):
        names = fs if female else ms
        grp[[i for i, sc in enumerate(g["superclass"]) if sc in names]] = k + 1
    role = np.zeros(g["n"], np.uint8)
    role[[i for v in g["input_groups"].values() for i in v]] = 1
    role[[i for v in g["output_groups"].values() for i in v]] = 2
    role[[i for v in g.get("audio_groups", {}).values() for i in v]] = 3  # hearing (Johnston's organ)
    meta = {"n": g["n"], "groups": [x[0] for x in SUPERCLASS_GROUPS], "estimated": int(g["pos_estimated"].sum()),
            "edges": int(len(g["pre"]))}
    return pos.astype(np.float32).tobytes() + grp.tobytes() + role.tobytes(), meta


def brain_edges(q):
    """Binary: Uint32 pre, Uint32 post, Float32 weight, Float32 learned scale for the selected subset of edges."""
    import numpy as np
    torch = _torch()
    pre, post, w, scale, info = _edges_for(q.get("model", "raw:female_full"))
    k = max(1000, min(int(q.get("k", 30000) or 30000), 200000))
    mode = q.get("select", "strongest")
    if mode == "random":
        idx = torch.randperm(len(w), generator=torch.Generator().manual_seed(0))[:k]
    elif mode == "changed":
        idx = torch.topk(torch.log(scale).abs(), min(k, len(w))).indices
    else:
        idx = torch.topk(w.abs(), min(k, len(w))).indices
    a = w.abs()
    meta = {"total_edges": int(len(w)), "shown": int(len(idx)), **info,
            "w_abs_p50": float(a.median()), "w_abs_p99": float(torch.quantile(a[torch.randperm(len(a))[:200000]], 0.99)),
            "w_abs_max": float(a.max()), "inhibitory_frac": float((w < 0).float().mean()),
            "scale_min": float(scale.min()), "scale_max": float(scale.max()), "learned": bool((scale != 1).any())}
    body = (pre[idx].numpy().astype(np.uint32).tobytes() + post[idx].numpy().astype(np.uint32).tobytes()
            + w[idx].numpy().astype(np.float32).tobytes() + scale[idx].numpy().astype(np.float32).tobytes())
    return body, meta


# ---------- Chat: talk to any checkpoint ----------
import threading

_chat_lock = threading.Lock()
_chat_models = collections.OrderedDict()


def _chat_model(model_id):
    """Load (and cache, up to 2) a checkpoint for generation. Imports the training code lazily."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, ".."))
    from mougia_train.sample_checkpoint import load_model
    if model_id in _chat_models:
        _chat_models.move_to_end(model_id)
        return _chat_models[model_id]
    run, f = model_id[5:].split("/", 1)
    path = os.path.join(RUNS, os.path.basename(run), "checkpoints", os.path.basename(f))
    model, tok, ck = load_model(path)
    _chat_models[model_id] = (model, tok, ck)
    while len(_chat_models) > 2:
        _, (m, _, _) = _chat_models.popitem(last=False)
        del m
        _torch().cuda.empty_cache()
    return _chat_models[model_id]


def chat_models(q):
    models = [m for m in brain_models(q)["models"] if m["id"].startswith("ckpt:")]
    # Fine-tuned LLM baselines: LoRA adapter folders under runs/<run>/checkpoints/<tag>/
    for _, run, cfg in reversed(run_dirs()):
        if cfg.get("kind") != "llm_baseline":
            continue
        ck = os.path.join(RUNS, run, "checkpoints")
        for tag in sorted(os.listdir(ck)) if os.path.isdir(ck) else []:
            if os.path.exists(os.path.join(ck, tag, "adapter_config.json")):
                models.append({"id": f"llm:{run}/{tag}", "group": cfg.get("label") or run, "graph": None, "label": tag,
                               "time": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(os.path.join(ck, tag))))})
    return {"models": models}


def _llm_model(model_id):
    """Base model in 4-bit + the run's LoRA adapter (needs the .venv-llm environment and ~6 GB of free GPU)."""
    if model_id in _chat_models:
        _chat_models.move_to_end(model_id)
        return _chat_models[model_id]
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    run, tag = model_id[4:].split("/", 1)
    cfg = read_json(os.path.join(RUNS, os.path.basename(run), "config.json")) or {}
    base = cfg.get("args", {}).get("model")
    tk = AutoTokenizer.from_pretrained(base)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    m = AutoModelForCausalLM.from_pretrained(base, device_map={"": 0}, torch_dtype=torch.bfloat16,
                                             quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                                    bnb_4bit_compute_dtype=torch.bfloat16))
    m = PeftModel.from_pretrained(m, os.path.join(RUNS, os.path.basename(run), "checkpoints", os.path.basename(tag))).eval()
    for k in list(_chat_models):  # one big model at a time
        del _chat_models[k]
    torch.cuda.empty_cache()
    _chat_models[model_id] = (m, tk, {"step": None, "args": {"sex": None}, "llm": True})
    return _chat_models[model_id]


def chat(body):
    import sys
    if os.path.join(ROOT, "..") not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, ".."))
    from mougia_train.generate import reply as gen_reply
    model_id = body.get("model", "")
    text = (body.get("text") or "").strip()[:400]
    if not (model_id.startswith("ckpt:") or model_id.startswith("llm:")) or not text:
        raise ValueError("pick a checkpoint and write a couplet")
    mode = body.get("mode") if body.get("mode") in ("verse", "raw") else "verse"
    style = body.get("style") if body.get("style") in ("duel", "cont") else "duel"
    temperature = min(1.5, max(0.2, float(body.get("temperature", 0.8))))
    t0 = time.time()
    with _chat_lock:
        if model_id.startswith("llm:"):
            from mougia_train.llm_finetune import llm_reply
            model, tok, ck = _llm_model(model_id)
            out = llm_reply(model, tok, text, mode=mode, n=16, temperature=temperature, kind=style)
        else:
            model, tok, ck = _chat_model(model_id)
            out = gen_reply(model, tok, text, mode=mode, n=16, temperature=temperature, style=style)
    args = ck.get("args", {})
    return {"reply": out["reply"], "scores": out["scores"], "logp": out["logp"], "candidates": out["candidates"][:6],
            "mode": mode, "style": style, "temperature": temperature, "step": ck.get("step"),
            "poet": "Krikri-8B" if ck.get("llm") else POET.get(args.get("sex"), "?"), "seconds": round(time.time() - t0, 2)}


POST_ROUTES = {"/api/chat": chat}

BINARY_ROUTES = {"/api/brain/neurons": brain_neurons, "/api/brain/edges": brain_edges}

ROUTES = {"/api/chat/models": chat_models, "/api/brain/models": brain_models, "/api/runs": list_runs, "/api/run_series": run_series, "/api/run_samples": run_samples,
          "/api/overview": lambda q: overview(), "/api/threads": list_threads, "/api/thread": get_thread,
          "/api/search": search, "/api/pairs": list_pairs}
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".svg": "image/svg+xml"}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            self._get()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # never drop the connection: tell the page what went wrong
            import traceback
            traceback.print_exc()
            try:
                self.send(500, "application/json; charset=utf-8", json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())
            except Exception:
                pass

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if url.path not in POST_ROUTES or n > 20000:
                return self.send(404, "application/json", b'{"error": "not found"}')
            body = json.loads(self.rfile.read(n) or b"{}")
            out = POST_ROUTES[url.path](body)
            self.send(200, "application/json; charset=utf-8", json.dumps(out, ensure_ascii=False).encode())
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValueError as e:
            self.send(400, "application/json; charset=utf-8", json.dumps({"error": str(e)}).encode())
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send(500, "application/json; charset=utf-8", json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())

    def _get(self):
        url = urllib.parse.urlparse(self.path)
        if url.path in BINARY_ROUTES:
            q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
            try:
                body, meta = BINARY_ROUTES[url.path](q)
            except (FileNotFoundError, KeyError, ValueError) as e:
                return self.send(404, "application/json", json.dumps({"error": str(e)}).encode())
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Meta", json.dumps(meta))
            self.send_header("Access-Control-Expose-Headers", "X-Meta")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
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
