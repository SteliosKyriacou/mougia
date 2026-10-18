"""Generic Blogger scraper via JSON feeds (posts + comments).
usage: python3 -I blogger.py HOST SITE MODE [label_regex]
MODE: couplets | poems | auto (auto: posts whose title/labels match tsiattista/distich regex -> couplets, else poems)
Every post is saved raw (feed json pages in html/). Records:
  kind=couplet : a 2-line (or 4 short line) unit; position = index within post; thread_id = post id
  kind=poem    : full verse text of a post (only verse-like lines kept in text; raw_lines = all lines)
  kind=comment : a blog comment containing verse (reply_to = parent comment id if threaded)
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

host, SITE, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
LABEL_RE = re.compile(sys.argv[4] if len(sys.argv) > 4 else r"τσια|τσιά|τσα[τ]+ι|δίστιχ|διστιχ|ΤΣΙΑ|ΔΙΣΤΙΧ|ΤΣΙΆ", re.I)
d = sitedir(SITE)


def feed_all(kind):
    out, start = [], 1
    while True:
        url = f"https://{host}/feeds/{kind}/default?alt=json&max-results=150&start-index={start}"
        t = get(url, SITE, ext=".json", delay=1.0)
        if not t:
            break
        try:
            f = json.loads(t)["feed"]
        except Exception:
            break
        es = f.get("entry", [])
        out += es
        total = int(f.get("openSearch$totalResults", {}).get("$t", 0))
        if not es or start + len(es) > total:
            break
        start += len(es)
    return out


def link_of(e):
    for l in e.get("link", []):
        if l.get("rel") == "alternate":
            return l["href"]
    return None


posts = feed_all("posts")
print(host, "posts:", len(posts))
recs = []
for e in posts:
    pid = e["id"]["$t"].split("post-")[-1]
    title = e["title"]["$t"]
    labels = [c["term"] for c in e.get("category", [])]
    html = e.get("content", e.get("summary", {})).get("$t", "")
    lines = html_lines(html)
    glines = [l for l in lines if greek_ratio(l) > 0.7]
    url = link_of(e)
    author = (e.get("author") or [{}])[0].get("name", {}).get("$t")
    date = e.get("published", {}).get("$t")
    mode = MODE
    if MODE == "auto":
        mode = "couplets" if (LABEL_RE.search(title) or any(LABEL_RE.search(x) for x in labels)) else "poems"
    base = {"url": url, "site": SITE, "post_title": title, "labels": labels, "blog_author": author, "date": date,
            "thread_id": f"post-{pid}"}
    if mode in ("couplets", "both"):
        us = units_from_lines(lines)
        authors = {u["author"] for u in us if u["author"]}
        duel = len(authors) >= 2
        prev_seg, seg_first = None, {}
        for i, u in enumerate(us):
            rec = {"kind": "couplet", "text": u["text"], **base, "position": i, "n_couplets_in_post": len(us),
                   "raw_lines": u["seg_lines"], "verses": u["verses"], "rhyme_ok": u["rhyme_ok"],
                   "segment": u["seg"], "author": u["author"], "id": f"{SITE}:{pid}:{i}", "cy_score": u["cy_score"]}
            if duel:
                seg_first.setdefault(u["seg"], rec["id"])
                # reply_to = first couplet of previous segment (turn) in the duel
                prevs = [s for s in seg_first if s < u["seg"]]
                rec["reply_to"] = seg_first[max(prevs)] if prevs and seg_first[u["seg"]] == rec["id"] else None
                rec["duel"] = True
            recs.append(rec)
    if mode in ("poems", "both"):
        verse = [l for l in glines if is_verse(l, 5, 21)]
        if len(verse) >= 2:
            recs.append({"kind": "poem", "text": "\n".join(verse), **base, "position": 0, "raw_lines": lines,
                         "n_verse_lines": len(verse)})

comments = feed_all("comments")
print(host, "comments:", len(comments))
comments.sort(key=lambda e: e.get("published", {}).get("$t", ""))
pos = {}
for e in comments:
    html = e.get("content", e.get("summary", {})).get("$t", "")
    lines = [l for l in html_lines(html) if l]
    if not lines or greek_ratio(" ".join(lines)) < 0.5:
        continue
    cid = e["id"]["$t"].split("post-")[-1]
    post_id = None
    for l in e.get("link", []):
        m = re.search(r"/posts/default/(\d+)", l.get("href", ""))
        if m: post_id = m.group(1)
    reply_to = None
    for l in e.get("link", []):
        if l.get("rel") == "related":
            reply_to = l["href"].rsplit("/", 1)[-1]
    ir = e.get("thr$in-reply-to", {})
    if not post_id and ir.get("href"):
        post_id = ir["href"]
    k = post_id or "?"
    pos[k] = pos.get(k, -1) + 1
    cbase = {"url": link_of(e) or ir.get("href"), "site": SITE,
             "author": (e.get("author") or [{}])[0].get("name", {}).get("$t"), "date": e.get("published", {}).get("$t"),
             "thread_id": f"post-{k}", "position": pos[k], "comment_id": cid, "reply_to": reply_to, "source_type": "comment"}
    us = [u for u in units_from_lines(lines) if u["rhyme_ok"]]
    if us:
        for j, u in enumerate(us):
            recs.append({"kind": "couplet", "text": u["text"], "verses": u["verses"], "rhyme_ok": True, "cy_score": u["cy_score"],
                         **cbase, "pos_in_comment": j, "raw_lines": u["seg_lines"]})
    else:
        recs.append({"kind": "comment", "text": "\n".join(lines), **cbase, "raw_lines": lines})
write_jsonl(os.path.join(d, "extracted.jsonl"), recs)
from collections import Counter
print(Counter(r["kind"] for r in recs))
