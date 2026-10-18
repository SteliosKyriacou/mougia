"""Word counts + sentence-level overlap between Cypriot text files (in given order).
Usage: python3 -I overlap_stats.py file1 file2 ..."""
import sys, re, unicodedata

def norm(l):
    l = unicodedata.normalize("NFD", l.lower())
    l = "".join(c for c in l if not unicodedata.combining(c))
    return re.sub(r"[^\w]+", " ", l).strip()

seen = set(); total = 0
for path in sys.argv[1:]:
    words = new = 0
    for raw in open(path, encoding="utf-8", errors="replace"):
        for sent in re.split(r"(?<=[.!;?·])\s+", raw):
            w = len(sent.split()); words += w
            n = norm(sent)
            if len(n) >= 15 and n not in seen:
                seen.add(n); new += w
    total += new
    print(f"{words:>9} words {new:>9} new-unique  {path}")
print("cumulative unique words (sentences>=15 chars):", total)
