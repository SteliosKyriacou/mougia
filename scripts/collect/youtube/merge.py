"""Merge extracted_text.jsonl + extracted_whisper.jsonl (+ extracted_ytasr.jsonl)
into data/old/raw/youtube/extracted.jsonl and print a summary."""
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "data/old/raw/youtube"
recs = []
for name in ["extracted_text.jsonl", "extracted_whisper.jsonl", "extracted_subtitle.jsonl"]:
    p = OUT / name
    if p.exists():
        rs = [json.loads(l) for l in open(p)]
        if name == "extracted_subtitle.jsonl":  # auto-captions: keep only rhyming pairs
            rs = [r for r in rs if r.get("rhyme_ok")]
        recs += rs
with open(OUT / "extracted.jsonl", "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(recs), Counter(r["source_kind"] for r in recs))
print("conf>=0.75:", Counter(r["source_kind"] for r in recs if r["confidence"] >= 0.75))
print("reply-ordered (reply_to_turn set):", sum(r.get("reply_to_turn") is not None for r in recs))
