"""Convert the Tsiattista Muse gold set (Almagesth/tsiattista-dataset) into extracted.jsonl couplet records."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from greek_meter import syllables  # noqa: E402

src = sys.argv[1]  # .../corpus_tsiattista_muse/repo/data/tsiattista.json
out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(src))), "extracted.jsonl")
poems = json.load(open(src, encoding="utf-8"))
with open(out, "w", encoding="utf-8") as f:
    for p in poems:
        text = p.get("text") or ""
        text = "\n".join(text) if isinstance(text, list) else text
        short = [l.strip() for l in text.split("\n") if l.strip()]
        # Join 8+7 half-lines back into 15-syllable verses.
        lines, buf = [], ""
        for l in short:
            buf = f"{buf} {l}".strip()
            if syllables(buf)[1] >= 13:
                lines.append(buf)
                buf = ""
        for i in range(0, len(lines) - 1, 2):
            f.write(json.dumps({"text": "\n".join(lines[i:i + 2]), "url": "https://github.com/Almagesth/tsiattista-dataset",
                                "poet": p.get("author"), "category": p.get("theme"), "category_norm": "tsiattista",
                                "dialect": "cypriot", "thread_id": f"muse-{p.get('id')}", "position": i // 2}, ensure_ascii=False) + "\n")
print(out)
