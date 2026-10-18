"""Search a WordPress site via REST API for tsiattista-related posts and extract rhyming couplets.
usage: python3 -I wp_search.py BASE SITE [term ...]
Saves every matching post's JSON page; keeps couplets (rhyme_ok or not; downstream filters)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
base, SITE = sys.argv[1].rstrip("/"), sys.argv[2]
terms = sys.argv[3:] or ["τσιαττιστά", "τσιατιστά", "τσιαττιστό", "τσιατιστό", "τσιάττισμα", "ποιητάρης", "ποιητάρηδες", "δίστιχα"]
d = sitedir(SITE)
posts = {}
for term in terms:
    page = 1
    while True:
        u = f"{base}/wp-json/wp/v2/posts?search={urllib.parse.quote(term)}&per_page=50&page={page}&_fields=id,link,date,title,content"
        t = get(u, SITE, ext=".json", delay=1.5)
        if not t:
            break
        try:
            js = json.loads(t)
        except Exception:
            break
        if not isinstance(js, list) or not js:
            break
        for p in js:
            posts[p["id"]] = p
        print(term, page, len(js), "total posts", len(posts))
        if len(js) < 50:
            break
        page += 1
recs = []
for pid, p in sorted(posts.items()):
    lines = html_lines(p["content"]["rendered"])
    us = units_from_lines(lines)
    title = BeautifulSoup(p["title"]["rendered"], "html.parser").get_text()
    for i, u in enumerate(us):
        recs.append({"kind": "couplet", "text": u["text"], "verses": u["verses"], "rhyme_ok": u["rhyme_ok"],
                     "cy_score": u["cy_score"], "url": p["link"], "site": SITE, "post_title": title, "date": p["date"],
                     "thread_id": f"post-{pid}", "position": i, "segment": u["seg"], "author": u["author"],
                     "raw_lines": u["seg_lines"]})
write_jsonl(os.path.join(d, "extracted.jsonl"), recs)
print("rhyming:", sum(r["rhyme_ok"] for r in recs))
