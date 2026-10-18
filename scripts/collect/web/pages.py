"""Extract couplets from individual article/blog pages.
usage: python3 -I pages.py LISTFILE      (lines: SITE URL [css-selector])
Writes data/raw/web_<SITE>/extracted.jsonl per site (all URLs of that site together)."""
import sys, os, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
SEL = [".post-body", ".entry-content", "article .content", "article", ".article-body", ".td-post-content", "#content", "main"]
by = defaultdict(list)
for ln in open(sys.argv[1]):
    if not ln.strip() or ln.startswith("#"): continue
    p = ln.split()
    by[p[0]].append((p[1], p[2] if len(p) > 2 else None))
for site, items in by.items():
    recs = []
    for url, sel in items:
        t = get(url, site)
        if not t:
            continue
        soup = BeautifulSoup(t, "html.parser")
        for s in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]): s.decompose()
        nodes = soup.select(sel) if sel else []
        if not nodes:
            for s in SEL:
                nodes = soup.select(s)
                if nodes: break
        nodes = nodes or [soup.body or soup]
        title = (soup.title.get_text().strip() if soup.title else None)
        pos = 0
        for node in nodes:
            for u in units_from_lines(html_lines(str(node))):
                recs.append({"kind": "couplet", "text": u["text"], "verses": u["verses"], "rhyme_ok": u["rhyme_ok"],
                             "cy_score": u["cy_score"], "url": url, "site": site, "post_title": title,
                             "thread_id": url, "position": pos, "segment": u["seg"], "author": u["author"],
                             "raw_lines": u["seg_lines"]})
                pos += 1
    write_jsonl(os.path.join(sitedir(site), "extracted.jsonl"), recs)
    print(site, "rhyming", sum(r["rhyme_ok"] for r in recs), "/", len(recs))
