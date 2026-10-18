"""Scrape tsiatista.wordpress.com: all posts, pages, comments via WP.com public API."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

SITE = "tsiatista_wordpress"
API = "https://public-api.wordpress.com/rest/v1.1/sites/tsiatista.wordpress.com"
d = sitedir(SITE)

comments = []
page = 1
while True:
    t = get(f"{API}/comments?number=100&page={page}&order=ASC", SITE, ext=".json")
    js = json.loads(t)
    comments += js["comments"]
    if len(js["comments"]) < 100:
        break
    page += 1
posts = []
for typ in ("post", "page"):
    t = get(f"{API}/posts?number=100&type={typ}&status=publish", SITE, ext=".json")
    posts += json.loads(t)["posts"]
# also save the front page html
get("https://tsiatista.wordpress.com/", SITE)

recs = []
for p in posts:
    lines = [l for l in html_lines(p["content"]) if greek_ratio(l) > 0.8]
    for i, c in enumerate(group_couplets(lines)):
        recs.append({"kind": "couplet", "text": "\n".join(c), "url": p["URL"], "site": SITE, "author": p["author"].get("name"),
                     "date": p["date"], "thread_id": f"post-{p['ID']}", "position": i, "raw_lines": c,
                     "source_type": "post"})
# thread position: order by date within post
comments.sort(key=lambda c: (c["post"]["ID"], c["date"]))
pos = {}
for c in comments:
    lines = html_lines(c["content"])
    k = c["post"]["ID"]
    pos[k] = pos.get(k, -1) + 1
    base = {"url": c["URL"], "site": SITE, "author": c["author"].get("name") or None, "date": c["date"],
            "thread_id": f"post-{k}", "position": pos[k], "comment_id": c["ID"],
            "reply_to": (c["parent"]["ID"] if c["parent"] else None), "source_type": "comment"}
    us = units_from_lines(lines)
    if us:
        for j, u in enumerate(us):
            recs.append({"kind": "couplet", "text": u["text"], "verses": u["verses"], "rhyme_ok": u["rhyme_ok"],
                         "cy_score": u["cy_score"], **base, "pos_in_comment": j, "raw_lines": u["seg_lines"],
                         "comment_text": "\n".join(lines)})
    else:  # e.g. single-line couplets the reflow could not split, greeklish
        recs.append({"kind": "comment", "text": "\n".join(lines), **base, "raw_lines": lines})
write_jsonl(os.path.join(d, "extracted.jsonl"), recs)
