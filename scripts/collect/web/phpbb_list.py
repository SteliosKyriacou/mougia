"""List phpBB forum topics whose titles match a regex. usage: python3 -I phpbb_list.py SITE BASE FORUM_ID REGEX MAXPAGES"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
site, base, fid, rx, maxp = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3], re.compile(sys.argv[4], re.I), int(sys.argv[5])
seen = {}
for p in range(maxp):
    t = get(f"{base}/viewforum.php?f={fid}&start={p*25}", site, delay=1.5)
    if not t: break
    soup = BeautifulSoup(t, "html.parser")
    a = soup.select("a.topictitle")
    if not a: break
    new = 0
    for x in a:
        m = re.search(r"t=(\d+)", x["href"])
        if m and m.group(1) not in seen:
            seen[m.group(1)] = x.get_text(); new += 1
    if not new: break
for k, v in seen.items():
    if rx.search(v): print(k, v)
print("topics scanned", len(seen), file=sys.stderr)
