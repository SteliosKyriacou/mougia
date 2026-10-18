"""Fetch a URL into a site dir and print the Greek text lines of the main content (for inspection).
usage: python3 -I peek.py SITE URL [css_selector]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
site, url = sys.argv[1], sys.argv[2]
sel = sys.argv[3] if len(sys.argv) > 3 else None
t = get(url, site)
if t is None:
    sys.exit(1)
soup = BeautifulSoup(t, "html.parser")
for s in soup(["script", "style", "nav", "footer", "header", "noscript"]):
    s.decompose()
node = soup.select_one(sel) if sel else (soup.find("article") or soup.body)
lines = html_lines(str(node))
for l in lines:
    if greek_ratio(l) > 0.6 or not sel:
        print(f"{approx_syll(l):3d} | {l}")
