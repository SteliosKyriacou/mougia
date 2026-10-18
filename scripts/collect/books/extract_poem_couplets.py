"""Poem-mode couplet extractor for clean verse texts (e.g. Wikisource, lyric booklets).
Within each stanza block (blank-line separated) lines are paired in order (trying offset 0 and 1);
a pair is kept when both lines have 12-18 syllables and they rhyme (accented tail or last-two-vowel key).
Usage: python3 -I extract_poem_couplets.py FILE --source S --url U [--min-dialect 1] [--title T]"""
import sys, json, argparse, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("ec", os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract_couplets.py"))
src = open(spec.origin, encoding="utf-8").read().replace("\nmain()\n", "\n")
ec = {}; exec(compile(src, spec.origin, "exec"), ec)
syll, accented_tail, rhyme_key, last_word, clean, greekish, dialect_score = (ec[k] for k in
    ["syllables", "accented_tail", "rhyme_key", "last_word", "clean", "greekish", "dialect_score"])

import re
def _norm_tail(t):
    t = re.sub(r"[νσ]$", "", t or "")
    return t
def _last_vowel(t):
    m = re.findall(r"[αειου]", t or ""); return m[-1] if m else ""
def rhymes(a, b, lenient=True):
    """strict: same accented tail / last-two-vowel key; lenient (folk assonance): same final vowel and
    same stress position counted from the end (final -ν/-ς ignored)."""
    if last_word(a) == last_word(b) or not last_word(a): return False
    ta, tb = accented_tail(a), accented_tail(b)
    if ta and tb and _norm_tail(ta) == _norm_tail(tb): return True
    if rhyme_key(a) == rhyme_key(b) and len(rhyme_key(a)) >= 2: return True
    if not lenient or not ta or not tb: return False
    na, nb = _norm_tail(ta), _norm_tail(tb)
    va = len(re.findall(r"[αειου]+", na)); vb = len(re.findall(r"[αειου]+", nb))
    return va == vb and _last_vowel(na) == _last_vowel(nb)

def verse(l):
    letters = [c for c in l if c.isalpha()]
    g = sum(1 for c in letters if "\u0370" <= c <= "\u03ff" or "\u1f00" <= c <= "\u1fff")
    return len(letters) >= 15 and g / len(letters) > 0.9 and 12 <= syll(l) <= 18

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--source", required=True)
    ap.add_argument("--url", default=""); ap.add_argument("--title", default=""); ap.add_argument("--min-dialect", type=int, default=0)
    ap.add_argument("--no-blocks", action="store_true", help="ignore blank lines; greedy sequential pairing over the whole text")
    ap.add_argument("--ocr", action="store_true", help="noisy OCR: drop non-verse lines first, one block per page")
    a = ap.parse_args()
    raw = open(a.path, encoding="utf-8").read()
    if a.ocr:
        def okline(l):
            letters = [c for c in l if c.isalpha()]
            if len(letters) < 12: return False
            g = sum(1 for c in letters if "\u0370" <= c <= "\u03ff" or "\u1f00" <= c <= "\u1fff")
            return g / len(letters) > 0.85 and syll(l) >= 10
        pages = raw.split("\f")
        raw = "\n\n".join("\n".join(clean(l) for l in pg.split("\n") if okline(clean(l))) for pg in pages)
    raw = raw.replace("\f", "\n\n")
    blocks, cur = [], []
    for l in raw.split("\n"):
        l = clean(l)
        if not l: 
            if cur: blocks.append(cur); cur = []
        else: cur.append(l)
    if cur: blocks.append(cur)
    if a.no_blocks:
        flat = [l for b in blocks for l in b]
        greedy, i = [], 0
        while i < len(flat) - 1:
            if verse(flat[i]) and verse(flat[i + 1]) and rhymes(flat[i], flat[i + 1]):
                greedy.append((flat[i], flat[i + 1])); i += 2
            else: i += 1
        blocks = [None]
    n = 0
    for b in blocks:
        if b is None:
            best = greedy
        else:
          best = []
          for off in (0, 1):
            pairs = [(b[i], b[i + 1]) for i in range(off, len(b) - 1, 2)]
            good = [p for p in pairs if verse(p[0]) and verse(p[1]) and rhymes(*p)]
            if len(good) > len(best): best = good
        for l1, l2 in best:
            t = l1 + "\n" + l2
            if dialect_score(t) < a.min_dialect: continue
            print(json.dumps({"text": t, "source": a.source, "url": a.url, "section": a.title, "ocr_confidence": "high",
                              "syllables": [syll(l1), syll(l2)], "dialect_score": dialect_score(t)}, ensure_ascii=False)); n += 1
    print(n, file=sys.stderr)
main()
