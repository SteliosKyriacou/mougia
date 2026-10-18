"""Merge visual transcriptions (part files) of Sakellarios vol.2 into extracted.jsonl + text.txt.
Usage: python3 -I build_sakellarios.py part1.jsonl part2.jsonl part3.jsonl"""
import json, sys
OUT = "/home/stelios/repos/mougia/data/raw/book_sakellarios1891"
URL = "https://archive.org/details/takypriakatoigeg02sakeuoft"
recs = []
for p in sys.argv[1:]:
    for line in open(p, encoding="utf-8"):
        if line.strip(): recs.append(json.loads(line))
order = {"disticha": 0, "tetrasticha": 1, "eptastichon": 2, "disticha_appendix": 3}
for r in recs: r.setdefault("section", "disticha")
recs.sort(key=lambda r: (order[r["section"]], r["num"]))
with open(f"{OUT}/extracted.jsonl", "w", encoding="utf-8") as f, open(f"{OUT}/text.txt", "w", encoding="utf-8") as t:
    for r in recs:
        o = {"text": r["text"], "source": "Sakellarios, Τα Κυπριακά, τ. Β΄ (1891)", "url": f"{URL}/page/n{r['leaf']}",
             "page": r.get("printed_page"), "leaf": r["leaf"], "num": r["num"], "section": r["section"],
             "ocr_confidence": r.get("ocr_confidence", "high"), "method": "visual transcription of page scans (Claude), tesseract grc as aid"}
        if r.get("notes"): o["notes"] = r["notes"]
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
        t.write(f"[{r['section']} {r['num']} leaf {r['leaf']}]\n{r['text']}\n\n")
from collections import Counter
print(len(recs), Counter(r["section"] for r in recs), Counter(r.get("ocr_confidence") for r in recs))
print(Counter(r["text"].count("\n") + 1 for r in recs))
