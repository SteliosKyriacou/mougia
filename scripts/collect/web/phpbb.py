"""phpBB3 topic scraper. usage: python3 -I phpbb.py SITE BASE TOPIC_ID [TOPIC_ID ...]
Couplet records keep: author (username), date string, thread position (post index), quoted user (reply_to_user)."""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
site, base, tids = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3:]
d = sitedir(site)
out_path = os.path.join(d, "extracted.jsonl")
recs = []
for tid in tids:
    start, pidx, seen = 0, 0, set()
    title = None
    while True:
        t = get(f"{base}/viewtopic.php?t={tid}&start={start}", site, delay=1.5)
        if not t: break
        soup = BeautifulSoup(t, "html.parser")
        title = title or (soup.select_one("h2.topic-title, h2") or soup.title).get_text().strip()
        posts = soup.select("div.post")
        ids = [p.get("id") for p in posts]
        if not posts or all(i in seen for i in ids): break
        for p in posts:
            if p.get("id") in seen: continue
            seen.add(p.get("id"))
            au = p.select_one(".author")
            user = au.select_one("a.username, a.username-coloured, strong") if au else None
            user = user.get_text().strip() if user else None
            date = None
            if au:
                tm = au.find("time")
                date = tm.get("datetime") if tm else re.sub(r".*»\s*", "", au.get_text()).strip()
            content = p.select_one(".content")
            if not content: continue
            quoted = []
            for q in content.find_all("blockquote"):
                c = q.find("cite")
                quoted.append(c.get_text().replace("έγραψε:", "").replace("wrote:", "").strip() if c else None)
                q.decompose()
            for i, u in enumerate(units_from_lines(html_lines(str(content)))):
                recs.append({"kind": "couplet", "text": u["text"], "verses": u["verses"], "rhyme_ok": u["rhyme_ok"],
                             "cy_score": u["cy_score"], "url": f"{base}/viewtopic.php?t={tid}#{p.get('id')}", "site": site,
                             "thread_title": title, "thread_id": f"topic-{tid}", "post_id": p.get("id"), "post_index": pidx,
                             "position": i, "author": user, "date": date, "reply_to_user": quoted[0] if quoted else None,
                             "segment": u["seg"], "raw_lines": u["seg_lines"]})
            pidx += 1
        start += len(posts)
# merge with existing records of other topics
old = []
if os.path.exists(out_path):
    old = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    old = [r for r in old if r.get("thread_id") not in {f"topic-{x}" for x in tids}]
write_jsonl(out_path, old + recs)
print("new", len(recs), "rhyming", sum(r["rhyme_ok"] for r in recs))
