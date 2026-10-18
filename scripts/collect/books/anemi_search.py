"""Search Anemi (anemi.lib.uoc.gr) simple search; print numbered result titles + srfile record ids."""
import sys, requests, re, urllib3
from bs4 import BeautifulSoup
urllib3.disable_warnings()
H = {"User-Agent": "Mozilla/5.0"}
BASE = "https://anemi.lib.uoc.gr"
def get(url, **kw):
    return requests.get(url, verify=False, timeout=120, headers=H, **kw)
def search(term, n=100):
    p = {"store_query": "1", "search_help": "list", "search_type": "simple", "display_mode": "overview", "wf_step": "init",
         "show_hidden": "0", "number": str(n), "cclterm1": term, "cclfield1": "term", "search_coll[metadata]": "1", "offset": "1", "dtab": "m"}
    return BeautifulSoup(get(BASE + "/search/", params=p).text, "html.parser")
def results(s):
    txt = s.get_text(" ", strip=True)
    items = re.findall(r"(\d+)\. Title: (.*?)(?=\s\d+\. Title: |\s*$)", txt)
    srf = []
    for a in s.find_all("a", href=True):
        m = re.search(r"srfile=(/metadata/[^&\"]+\.tkl)", a["href"])
        if m and m.group(1) not in srf: srf.append(m.group(1))
    return items, srf
if __name__ == "__main__":
    for t in sys.argv[1:]:
        items, srf = results(search(t))
        print("###", t, len(items), "items", len(srf), "srfiles")
        for n, ti in items: print(f"  {n}. {ti[:230]}")
