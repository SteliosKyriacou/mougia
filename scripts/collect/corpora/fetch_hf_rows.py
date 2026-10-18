"""Fetch text columns of a HF dataset via datasets-server /rows API (skips audio).
Usage: python3 -I fetch_hf_rows.py DATASET OUT.jsonl col1,col2 [config]"""
import sys, json, time, requests
ds, out, cols = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
cfg = sys.argv[4] if len(sys.argv) > 4 else "default"
S = requests.Session()
info = S.get(f"https://datasets-server.huggingface.co/splits?dataset={ds}").json()
with open(out, "w") as f:
    for sp in info["splits"]:
        if sp["config"] != cfg: continue
        split = sp["split"]; off = 0; total = None
        while total is None or off < total:
            for attempt in range(6):
                r = S.get("https://datasets-server.huggingface.co/rows",
                          params=dict(dataset=ds, config=cfg, split=split, offset=off, length=100))
                if r.status_code == 200: break
                time.sleep(3 * (attempt + 1))
            d = r.json(); total = d["num_rows_total"]
            for row in d["rows"]:
                f.write(json.dumps({"split": split, **{c: row["row"].get(c) for c in cols}}, ensure_ascii=False) + "\n")
            off += 100
        print(split, total, file=sys.stderr)
