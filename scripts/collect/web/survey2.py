"""Survey v2 using units_from_lines (reflow + strict rhyme). input lines: SITE URL"""
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
    us = units_from_lines(html_lines(str(soup.body or soup)))
    good = [u for u in us if u["rhyme_ok"]]
    print(f"{len(good):4d}/{len(us):4d} {site} {url[:100]}")
    for u in good[:2]: print("      e.g.", u["text"].replace("\n", " / ")[:160])
