"""Build the books-only corpus in data/books/: Cypriot 15-syllable verse from books, poem after poem.

Sources (books only): Sakellarios, Τα Κυπριακά (1891) — folk songs and distichs; Cypriot poets from Wikisource
(Michaelides, Lipertis …); the school anthology «Κείμενα Κυπριακής Λογοτεχνίας»; the Ττερλίκκας song booklet.
Left out: newspaper (Orthodox Herald) and thesis quotations, Standard-Greek / mixed poems (dialect labels from the
earlier reading), and anything not in the 15-syllable line: a poem is kept when ≥ 70% of its lines are 13–17
syllables, and then only those lines are kept, in order. Spelling: monotonic, Cypriot conjunction τζιαι
(see mougia_train/books_prepare.normalize).

Writes data/books/books.txt (one verse per line, a blank line between poems), data/books/poems.jsonl (same poems
with id/source/title), data/books/SOURCES.md.
"""
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from greek_meter import is_15_syllable  # noqa: E402
from mougia_train.books_prepare import normalize  # noqa: E402

OLD = os.path.join(ROOT, "data", "old", "processed", "books_v1")
OUT = os.path.join(ROOT, "data", "books")
SOURCES = {"sakellarios1891": "Σακελλάριος, Τα Κυπριακά, τόμ. Β΄ (1891) — δημώδη άσματα",
           "sakellarios1891-distichs": "Σακελλάριος, Τα Κυπριακά, τόμ. Β΄ (1891) — δίστιχα",
           "wikisource_cy": "Βικιθήκη — Κύπριοι ποιητές (Μιχαηλίδης, Λιπέρτης …)",
           "schcy_keimena": "Κείμενα Κυπριακής Λογοτεχνίας (σχολική ανθολογία)",
           "tterlikkas2002": "Ττερλίκκας — φυλλάδιο παραδοσιακών τραγουδιών (2002)"}
EXCLUDE_LINES = {"wiki-η-9η-ιουλίου-του-1821": [(0, 62)], "wiki-τα-προξενεία-της-αφροδίτης": [(0, 11), (48, 56)]}


def main():
    poems = [json.loads(l) for l in open(os.path.join(OLD, "poems.jsonl"), encoding="utf-8")]
    for p in map(json.loads, open(os.path.join(OLD, "extra_poems.jsonl"), encoding="utf-8")):
        p["lines"] = [normalize(l) for l in p["lines"] if l.strip()]
        poems.append(p)
    dialect = {r["poem_id"]: r["dialect"] for f in glob.glob(os.path.join(OLD, "annot", "*.jsonl"))
               for r in map(json.loads, open(f, encoding="utf-8"))}
    kept, stats = [], {"poems_in": 0, "not_book": 0, "not_cypriot": 0, "not_15_syllable": 0, "lines_dropped": 0}
    for p in poems:
        stats["poems_in"] += 1
        if p["source"] not in SOURCES:
            stats["not_book"] += 1
            continue
        if dialect.get(p["poem_id"], "cypriot" if p["source"] == "sakellarios1891-distichs" else None) != "cypriot":
            stats["not_cypriot"] += 1
            continue
        drop = {i for lo, hi in EXCLUDE_LINES.get(p["poem_id"], []) for i in range(lo, hi + 1)}
        lines = [l for i, l in enumerate(p["lines"]) if i not in drop and l.strip()]
        ok = [l for l in lines if is_15_syllable(l)]
        if not lines or len(ok) / len(lines) < 0.7:
            stats["not_15_syllable"] += 1
            continue
        stats["lines_dropped"] += len(lines) - len(ok)
        kept.append({"poem_id": p["poem_id"], "source": p["source"], "title": p.get("title"), "lines": ok})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "books.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join("\n".join(p["lines"]) for p in kept) + "\n")
    with open(os.path.join(OUT, "poems.jsonl"), "w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    per = {}
    for p in kept:
        per.setdefault(p["source"], [0, 0])
        per[p["source"]][0] += 1
        per[p["source"]][1] += len(p["lines"])
    stats.update(poems_kept=len(kept), lines_kept=sum(len(p["lines"]) for p in kept))
    with open(os.path.join(OUT, "SOURCES.md"), "w", encoding="utf-8") as f:
        f.write("# Books-only corpus — Cypriot 15-syllable verse\n\n`books.txt`: one verse per line, a blank line between poems.\n"
                "Built by `scripts/build_books_corpus.py` from the earlier book extraction (now in `data/old/`).\n\n"
                "| Source | Poems | Verses |\n|---|---|---|\n")
        for src, (n, l) in per.items():
            f.write(f"| {SOURCES[src]} | {n} | {l} |\n")
        f.write(f"\n{json.dumps(stats, ensure_ascii=False)}\n")
    print(json.dumps(stats, ensure_ascii=False), json.dumps(per, ensure_ascii=False))


if __name__ == "__main__":
    main()
