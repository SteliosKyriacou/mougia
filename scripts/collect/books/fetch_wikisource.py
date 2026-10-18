"""Fetch plain text of all main-namespace pages linked from given el.wikisource author pages.
Usage: python3 -I fetch_wikisource.py OUTDIR 'Συγγραφέας:...' [...]"""
import sys, os, re, json, requests, time
API = "https://el.wikisource.org/w/api.php"; H = {"User-Agent": "mougia-research/0.1 (dataset of Cypriot couplets)"}
out = os.path.join(sys.argv[1], "original"); os.makedirs(out, exist_ok=True)
index = []
def get_json(params):
    for k in range(8):
        resp = requests.get(API, params=params, headers=H, timeout=60)
        if resp.status_code == 200:
            try: return resp.json()
            except Exception: pass
        print("retry", resp.status_code, params.get("titles") or params.get("page"), file=sys.stderr)
        time.sleep(10 * (k + 1))
    return None
for author in sys.argv[2:]:
    j = get_json({"action": "query", "prop": "links", "titles": author, "pllimit": 500, "plnamespace": 0, "format": "json"})
    titles = [l["title"] for p in j["query"]["pages"].values() for l in p.get("links", [])]
    for t in titles:
        fn = re.sub(r"[^\w\-]+", "_", t)[:80] + ".txt"
        url = "https://el.wikisource.org/wiki/" + t.replace(" ", "_")
        if os.path.exists(os.path.join(out, fn)):
            index.append({"author": author, "title": t, "file": fn, "url": url}); continue
        r = get_json({"action": "parse", "page": t, "prop": "text", "format": "json", "redirects": 1})
        if not r or "parse" not in r: print("FAIL", t, file=sys.stderr); continue
        html = r["parse"]["text"]["*"]
        from bs4 import BeautifulSoup
        s = BeautifulSoup(html, "html.parser")
        for x in s.select(".mw-editsection, .noprint, sup.reference, .reference, table.headertemplate, #headertemplate, .ws-noexport"): x.decompose()
        txt = s.get_text("\n")
        txt = re.sub(r"\n\s*\n+", "\n\n", txt).strip()
        open(os.path.join(out, fn), "w", encoding="utf-8").write(txt)
        index.append({"author": author, "title": t, "file": fn, "url": url})
        time.sleep(4.0)
json.dump(index, open(os.path.join(sys.argv[1], "index.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(index))
