"""Query archive.org full-text search (be-api fts); print identifier | title | file | highlight snippets."""
import sys, requests
def fts(q, n=100):
    r = requests.get("https://be-api.us.archive.org/fts/v1/search", params={"q": q, "size": n}, timeout=90)
    r.raise_for_status(); return r.json()
if __name__ == "__main__":
    seen = {}
    for q in sys.argv[1:]:
        j = fts(q)
        print("###", q, j.get("hits", {}).get("total"))
        for h in j["hits"]["hits"]:
            f = h["fields"]; i = f["identifier"][0]
            if i in seen: continue
            seen[i] = 1
            hl = " || ".join(h.get("highlight", {}).get("text", [])[:2]).replace("\n", " ")
            print(f"  {i} | {f.get('meta_title',[''])[0][:80]} | {f.get('meta_year',[''])[0] if f.get('meta_year') else ''} | {hl[:250]}")
