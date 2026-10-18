"""Survey candidate URLs: fetch each (saving to web_<site>/html) and report #couplets found.
input lines: SITE URL"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
for ln in open(sys.argv[1]):
    if not ln.strip() or ln.startswith("#"): continue
    site, url = ln.split(None, 1); url = url.strip()
    try:
        t = get(url, site)
    except Exception as e:
        print("ERR", site, url, e); continue
    if not t: print("FAIL", site, url); continue
    soup = BeautifulSoup(t, "html.parser")
    for s in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]): s.decompose()
    lines = [l for l in html_lines(str(soup.body or soup)) if greek_ratio(l) > 0.8]
    cps = group_couplets(lines, need_rhyme=True)
    print(f"{len(cps):4d} {site} {url[:110]}")
    if cps: print("      e.g.", " / ".join(cps[0])[:160])
