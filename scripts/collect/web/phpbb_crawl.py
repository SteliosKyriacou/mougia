"""Full, resumable crawl of a phpBB3 forum, keeping only posts that contain verse.

usage: python3 -I phpbb_crawl.py SITE BASE "F1,F2,..." HOURS
  F list = forum ids in priority order; HOURS = time budget (stops cleanly, resumable).

Raw storage: each topic page is saved *trimmed* (topic title + the <div class="post"> elements only,
i.e. the original post HTML without site chrome) and gzipped: html/topics/t<tid>_s<start>.html.gz
Forum listing pages are saved trimmed too: html/listing/f<fid>_s<start>.html.gz
Progress: crawl_state.json (done topic ids). Records: crawl_couplets.jsonl (appended per topic).
A post is kept iff at least one reflowed unit rhymes (rhyme_ok); all units of a kept post are written.
"""
import sys, os, re, json, time, gzip, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import S, sitedir, html_lines, units_from_lines, BeautifulSoup

site, base, fids, hours = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3].split(","), float(sys.argv[4])
DELAY = 1.5
d = sitedir(site)
os.makedirs(os.path.join(d, "html", "topics"), exist_ok=True)
os.makedirs(os.path.join(d, "html", "listing"), exist_ok=True)
state_p = os.path.join(d, "crawl_state.json")
out_p = os.path.join(d, "crawl_couplets.jsonl")
state = json.load(open(state_p)) if os.path.exists(state_p) else {"done": [], "topics": {}, "pages": 0}
done = set(state["done"])
if os.environ.get("REEXTRACT") == "1":  # re-parse everything from cached pages (no refetch of saved pages)
    done = set()
    open(out_p, "w").close()
t_end = time.time() + hours * 3600
_last = [0.0]


def fetch(url):
    w = DELAY - (time.time() - _last[0])
    if w > 0:
        time.sleep(w)
    _last[0] = time.time()
    for a in range(4):
        try:
            r = S.get(url, timeout=40)
            if r.status_code == 200:
                r.encoding = "utf-8"
                return r.text
            print("HTTP", r.status_code, url, flush=True)
            if r.status_code in (403, 404):
                return None
        except Exception as e:
            print("ERR", e, url, flush=True)
        time.sleep(10 * (a + 1))
    return None


def save(path, html):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(html)


def load(path):
    if os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    return None


def list_forum(fid):
    topics, start = {}, 0
    while True:
        p = os.path.join(d, "html", "listing", f"f{fid}_s{start}.html.gz")
        t = load(p)
        if t is None:
            t = fetch(f"{base}/viewforum.php?f={fid}&start={start}")
            if t is None:
                break
            soup = BeautifulSoup(t, "html.parser")
            rows = soup.select("ul.topiclist.topics li.row, li.row")
            save(p, "\n".join(str(r) for r in rows))
            t = load(p)
        soup = BeautifulSoup(t, "html.parser")
        new = 0
        for row in soup.select("li.row"):
            a = row.select_one("a.topictitle")
            if not a:
                continue
            m = re.search(r"t=(\d+)", a["href"])
            if not m or m.group(1) in topics:
                continue
            po = row.select_one("dd.posts")
            try:
                replies = int(re.sub(r"\D", "", po.get_text().split()[0]))
            except Exception:
                replies = 0
            topics[m.group(1)] = {"title": a.get_text().strip(), "replies": replies, "forum": fid}
            new += 1
        if new == 0:
            break
        start += 25
    return topics


def crawl_topic(tid, meta):
    recs, start, pidx, seen = [], 0, 0, set()
    prev = None  # (post_id, author, had_verse)
    npages = max(1, math.ceil((meta["replies"] + 1) / 10))
    for _ in range(npages + 2):
        p = os.path.join(d, "html", "topics", f"t{tid}_s{start}.html.gz")
        t = load(p)
        if t is None:
            t = fetch(f"{base}/viewtopic.php?t={tid}&start={start}")
            if t is None:
                break
            soup = BeautifulSoup(t, "html.parser")
            h = soup.select_one("h2.topic-title, h2")
            posts = soup.select("div.post")
            save(p, (str(h) if h else "") + "\n" + "\n".join(str(x) for x in posts))
            state["pages"] += 1
            t = load(p)
        soup = BeautifulSoup(t, "html.parser")
        posts = soup.select("div.post")
        ids = [x.get("id") for x in posts]
        if not posts or all(i in seen for i in ids):
            break
        for po in posts:
            pid = po.get("id")
            if pid in seen:
                continue
            seen.add(pid)
            au = po.select_one(".author")
            user = au.select_one("a.username, a.username-coloured, strong") if au else None
            user = user.get_text().strip() if user else None
            date = None
            if au:
                tm = au.find("time")
                date = tm.get("datetime") if tm else re.sub(r".*»\s*", "", au.get_text()).strip()
            content = po.select_one(".content")
            if content is None:
                continue
            quoted, quoted_pid = [], None
            for q in content.find_all("blockquote"):
                c = q.find("cite")
                if c:
                    ca = c.find("a", href=re.compile(r"p=\d+"))
                    if ca and not quoted_pid:
                        quoted_pid = "p" + re.search(r"p=(\d+)", ca["href"]).group(1)
                    quoted.append(re.sub(r"\s*(έγραψε|wrote)\s*:?.*$", "", c.get_text(" ").strip(), flags=re.S).strip())
                q.decompose()
            us = units_from_lines(html_lines(str(content)))
            has_verse = any(u["rhyme_ok"] for u in us)
            if has_verse:
                for i, u in enumerate(us):
                    recs.append({"kind": "couplet", "text": u["text"], "verses": u["verses"], "rhyme_ok": u["rhyme_ok"],
                                 "cy_score": u["cy_score"], "url": f"{base}/viewtopic.php?t={tid}&start={start}#{pid}",
                                 "site": site, "thread_title": meta["title"], "forum_id": meta["forum"],
                                 "thread_id": f"topic-{tid}", "post_id": pid, "post_index": pidx, "position": i,
                                 "author": user, "date": date,
                                 "reply_to_user": quoted[0] if quoted else None, "reply_to_post_id": quoted_pid,
                                 "prev_post_id": prev[0] if prev else None, "prev_post_author": prev[1] if prev else None,
                                 "prev_post_has_verse": prev[2] if prev else None,
                                 "segment": u["seg"], "raw_lines": u["seg_lines"]})
            prev = (pid, user, has_verse)
            pidx += 1
        start += 10
    return recs


t0 = time.time()
for fid in fids:
    if time.time() > t_end:
        break
    topics = list_forum(fid)
    state["topics"].update(topics)
    print(f"forum {fid}: {len(topics)} topics, ~{sum(math.ceil((m['replies']+1)/10) for m in topics.values())} pages", flush=True)
    for tid, meta in topics.items():
        if tid in done:
            continue
        if time.time() > t_end:
            break
        recs = crawl_topic(tid, meta)
        with open(out_p, "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        done.add(tid)
        state["done"] = sorted(done)
        json.dump(state, open(state_p, "w"))
        if recs:
            print(f"  t{tid} {len(recs)} units ({sum(r['rhyme_ok'] for r in recs)} rhyming) {meta['title'][:50]}", flush=True)
print("STOP", "topics done", len(done), "pages fetched total", state["pages"], "elapsed h", round((time.time() - t0) / 3600, 2), flush=True)
