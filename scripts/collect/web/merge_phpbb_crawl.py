"""Merge crawl_couplets.jsonl (from phpbb_crawl.py) into extracted.jsonl of a site dir.
Crawl records supersede earlier per-topic records (phpbb.py) for the same thread_id.
usage: python3 -I merge_phpbb_crawl.py SITE"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import sitedir, write_jsonl
d = sitedir(sys.argv[1])
crawl = [json.loads(l) for l in open(os.path.join(d, "crawl_couplets.jsonl"), encoding="utf-8")]
tids = {r["thread_id"] for r in crawl}
old = [json.loads(l) for l in open(os.path.join(d, "extracted.jsonl"), encoding="utf-8")] if os.path.exists(os.path.join(d, "extracted.jsonl")) else []
old = [r for r in old if r.get("thread_id") not in tids]
seen, out = set(), []
for r in old + crawl:
    k = (r.get("post_id"), r.get("thread_id"), r.get("position"))
    if r.get("post_id") and k in seen:
        continue
    seen.add(k)
    out.append(r)
write_jsonl(os.path.join(d, "extracted.jsonl"), out)
print("kept old (topics not crawled):", len(old), "crawl:", len(crawl))
