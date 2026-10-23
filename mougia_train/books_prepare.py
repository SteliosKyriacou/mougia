"""Books-only dataset, step 1: split every book source into poems with numbered, normalised lines.

Writes data/processed/books_v1/poems.jsonl — one record per poem:
  {poem_id, source, title, author, lines: [normalised verse lines], original: [lines as printed]}
Normalisation (so Krikri learns *Cypriot*, not 19th-century Greek spelling):
  - polytonic → monotonic (breathings / iota subscript dropped, grave & circumflex → acute), ϐ → β;
  - the conjunction written καὶ / κὴ / και / κι (pronounced τζιαι in Cypriot) → τζιαι, capitalisation kept;
  - the original printed lines are kept alongside.
Sources handled here: Sakellarios 1891 (78 folk songs + 457 distichs), Cypriot Wikisource poems (Michaelides,
Lipertis, …), Orthodox Herald couplets. Mixed prose/verse books (school anthology, Ττερλίκκας booklet, thesis) are
extracted by hand-checked annotation into data/processed/books_v1/extra_poems.jsonl and merged by books_build.py.
"""
import glob
import json
import os
import re
import sys
import unicodedata

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "data", "raw")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from greek_meter import nuclei  # noqa: E402  (synizesis-aware syllables: γιά, μιά are one syllable)
OUT = os.path.join(ROOT, "data", "processed", "books_v1")

CONJ = re.compile(r"(?<![^\W\d_])(καί|και|κή|κη|κι|κί)(?![^\W\d_])", re.IGNORECASE)


def normalize(line):
    out = []
    for ch in unicodedata.normalize("NFD", line.replace("ϐ", "β")):
        if ch in "̯̓̔ͅ":
            continue
        out.append("́" if ch in "̀͂" else ch)
    t = unicodedata.normalize("NFC", "".join(out))
    t = re.sub(r"[ʼ'`΄´‘]", "’", t)
    t = CONJ.sub(lambda m: "Τζιαι" if m.group(0)[0].isupper() else "τζιαι", t)
    t = WORD.sub(lambda m: _unaccent_monosyllable(m.group(0)), t)
    return re.sub(r"\s+", " ", t).strip()


WORD = re.compile(r"[^\W\d_]+")
VOWEL_GROUP = re.compile(r"(?:αι|ει|οι|υι|ου|αυ|ευ|ηυ|[αεηιουω])", re.IGNORECASE)


def _unaccent_monosyllable(w):
    """Monotonic spelling: one-syllable words carry no accent (polytonic marked them with a grave: τὸ → το)."""
    if len(nuclei(w)[1]) == 1 and w.lower() not in ("ή", "πού", "πώς"):
        return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", w) if c != "\u0301"))
    return w


def poem(pid, source, title, lines, author=None):
    lines = [l for l in lines if l.strip() and not re.fullmatch(r"[\d\s.,–-]+", l.strip())]
    return {"poem_id": pid, "source": source, "title": title, "author": author,
            "lines": [normalize(l) for l in lines], "original": lines}


def sakellarios_songs():
    text = open(os.path.join(RAW, "book_sakellarios1891", "songs_text.txt"), encoding="utf-8").read()
    parts = re.split(r"^## ", text, flags=re.M)[1:]
    out = []
    for p in parts:
        head, _, body = p.partition("\n")
        m = re.match(r"\s*(\d+)\.?\s*(.*)", head)
        num, title = (m.group(1), m.group(2)) if m else (str(len(out) + 1), head)
        lines = [re.sub(r"\s*\(ἢ\)\s*$|\s*\d+\s*$", "", l).strip() for l in body.split("\n")]
        lines = [l for l in lines if l and not l.startswith(("#", ">"))]
        out.append(poem(f"sak-song-{int(num):02d}", "sakellarios1891", normalize(title.strip(" .")), lines,
                        "folk (Sakellarios 1891)"))
    return out


def sakellarios_distichs():
    out = []
    for path in sorted(glob.glob(os.path.join(RAW, "book_sakellarios1891", "transcription_parts", "sak_part*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            out.append(poem(f"sak-distich-{r['num']:03d}", "sakellarios1891-distichs", f"δίστιχο {r['num']}",
                            r["text"].split("\n"), "folk (Sakellarios 1891)"))
    return out


def wikisource():
    out = []
    for path in sorted(glob.glob(os.path.join(RAW, "book_wikisource_cy", "original", "*.txt"))):
        title = os.path.basename(path)[:-4].replace("_", " ")
        lines = [l.strip() for l in open(path, encoding="utf-8")]
        out.append(poem("wiki-" + re.sub(r"\W+", "-", title.lower()).strip("-")[:60], "wikisource_cy", title, lines))
    return out


def orthodox_herald():
    out = []
    for i, line in enumerate(open(os.path.join(RAW, "book_orthodox_herald", "extracted.jsonl"), encoding="utf-8")):
        r = json.loads(line)
        out.append(poem(f"oh-{i:03d}", "orthodox_herald", r.get("source") or "Ορθόδοξος Κήρυξ", r["text"].split("\n")))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    poems = sakellarios_songs() + sakellarios_distichs() + wikisource() + orthodox_herald()
    seen = {}
    for p in poems:  # variants share a number in the book (ἕτερον ᾆσμα…), distich numbering restarts per part
        n = seen.get(p["poem_id"], 0)
        seen[p["poem_id"]] = n + 1
        if n:
            p["poem_id"] += f"-{chr(ord('b') + n - 1)}"
    with open(os.path.join(OUT, "poems.jsonl"), "w", encoding="utf-8") as f:
        for p in poems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    by = {}
    for p in poems:
        by.setdefault(p["source"], [0, 0])
        by[p["source"]][0] += 1
        by[p["source"]][1] += len(p["lines"])
    print(json.dumps(by, ensure_ascii=False), "total poems", len(poems))


if __name__ == "__main__":
    main()
