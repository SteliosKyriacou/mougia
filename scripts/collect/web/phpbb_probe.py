"""Probe all topics of a phpBB forum: fetch page 1 of each topic and count rhyming dialect couplets.
usage: python3 -I phpbb_probe.py SITE BASE FORUM_ID MAXLISTPAGES  -> prints 'tid count title' for topics with >=2"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
site, base, fid, maxp = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3], int(sys.argv[4])
topics = {}
for p in range(maxp):
    t = get(f"{base}/viewforum.php?f={fid}&start={p*25}", site, delay=1.5)
    if not t:
        break
    a = BeautifulSoup(t, "html.parser").select("a.topictitle")
    new = 0
    for x in a:
        m = re.search(r"t=(\d+)", x["href"])
        if m and m.group(1) not in topics:
            topics[m.group(1)] = x.get_text()
            new += 1
    if not new:
        break
print("topics", len(topics), file=sys.stderr, flush=True)
for tid, title in topics.items():
    t = get(f"{base}/viewtopic.php?t={tid}&start=0", site, delay=1.5)
    if not t:
        continue
    soup = BeautifulSoup(t, "html.parser")
    n = 0
    for c in soup.select("div.post .content"):
        for q in c.find_all("blockquote"):
            q.decompose()
        n += sum(1 for u in units_from_lines(html_lines(str(c))) if u["rhyme_ok"] and dialect_of(u["text"]) == "cypriot")
    if n >= 2:
        print(tid, n, title, flush=True)
