"""Discourse forum topic scraper. usage: python3 -I discourse.py BASE SITE TOPIC_ID [TOPIC_ID...]
Records per couplet; also keeps the post (thread position, author, reply_to_post_number)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
base, SITE, tids = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3:]
d = sitedir(SITE)
recs = []
for tid in tids:
    t = json.loads(get(f"{base}/t/{tid}.json", SITE, ext=".json", delay=2))
    posts = t["post_stream"]["posts"]
    have = {p["id"] for p in posts}
    rest = [i for i in t["post_stream"]["stream"] if i not in have]
    for k in range(0, len(rest), 20):
        q = "&".join(f"post_ids[]={i}" for i in rest[k:k + 20])
        posts += json.loads(get(f"{base}/t/{tid}/posts.json?{q}", SITE, ext=".json", delay=2))["post_stream"]["posts"]
    posts.sort(key=lambda p: p["post_number"])
    for p in posts:
        # drop quoted blocks (they repeat other posts) but remember them
        soup = BeautifulSoup(p["cooked"], "html.parser")
        quotes = [q.extract() for q in soup.find_all("aside", class_="quote")]
        lines = html_lines(str(soup))
        us = units_from_lines(lines)
        for i, u in enumerate(us):
            recs.append({"kind": "couplet", "text": u["text"], "verses": u["verses"], "rhyme_ok": u["rhyme_ok"],
                         "url": f"{base}/t/{tid}/{p['post_number']}", "site": SITE, "thread_title": t["title"],
                         "author": p["username"], "date": p["created_at"], "thread_id": f"topic-{tid}",
                         "post_number": p["post_number"], "position": i,
                         "reply_to_post_number": p.get("reply_to_post_number"),
                         "quoted": [q.get("data-username") for q in quotes] or None,
                         "raw_lines": u["seg_lines"]})
write_jsonl(os.path.join(d, "extracted.jsonl"), recs)
