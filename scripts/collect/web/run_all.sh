#!/bin/bash
# Re-run all web extractors (uses cached raw pages in data/raw/web_*/html) and tag records.
cd "$(dirname "$0")"
python3 -I tsiatista_wordpress.py
python3 -I blogger.py kupriakatsiatismata.blogspot.com kupriakatsiatismata couplets
python3 -I blogger.py shapxar.blogspot.com shapxar auto
python3 -I blogger.py milomena.blogspot.com milomena couplets
python3 -I blogger.py kypriaki-dialektos.blogspot.com kypriaki_dialektos_blog both
python3 -I blogger.py poetryfromcyprus.blogspot.com poetryfromcyprus both
python3 -I pages.py pages_list.txt
python3 -I phpbb.py cyprushunting https://www.cyprushuntingmagazine.com/forum $(cat hunting_topics.txt)
[ -f ../../../data/raw/web_cyprushunting/crawl_couplets.jsonl ] && python3 -I merge_phpbb_crawl.py cyprushunting
python3 -I tag_records.py
