"""Heuristic extractor of rhyming ~15-syllable Greek couplets from plain text.
Usage: python3 -I extract_couplets.py text.txt --source NAME --url URL [--min 12 --max 18] [--page-sep '\f'] > out.jsonl
Pairs consecutive lines that (a) are mostly Greek, (b) have 12-18 vowel-group syllables, (c) rhyme on the
last syllable (accent-stripped). Emits records with ocr_confidence=medium (auto) and diagnostics."""
import sys, re, json, argparse, unicodedata

VOW = "αεηιουω"
def strip_acc(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("ς", "σ")

def syllables(line):
    s = re.sub(r"[^α-ω ]", "", strip_acc(line))
    s = re.sub(r"(αι|ει|οι|ου|υι|αυ|ευ|ηυ)", "A", s)
    s = re.sub(r"[ιυ](?=[αεηοωA])", "", s)   # glide (synizesis) approx
    return len(re.findall(r"[αεηιουωA]", s))

def last_word(line):
    ws = [re.sub(r"[^α-ω]", "", strip_acc(w)) for w in line.split()]
    ws = [w for w in ws if w]
    return ws[-1] if ws else ""

def rhyme_key(line):
    w = last_word(line)
    for x, y in {"ει": "ι", "οι": "ι", "υι": "ι", "αι": "ε", "η": "ι", "υ": "ι", "ω": "ο"}.items():
        w = w.replace(x, y)
    w = w.replace("σσ", "σ").replace("λλ", "λ").replace("νν", "ν")
    m = re.search(r"([αειου][^αειου]*[αειου][^αειου]*)$", w)   # last two vowels (≈ feminine/paroxytone rhyme)
    return m.group(1) if m else w

def accented_tail(line):
    ws = [w for w in line.split() if re.search(r"[\u0370-\u03ff\u1f00-\u1fff]", w)]
    if not ws: return None
    w = unicodedata.normalize("NFD", ws[-1].lower())
    letters = []  # (base, accented?)
    for c in w:
        if unicodedata.category(c) == "Mn":
            if letters and c in "\u0301\u0342\u0300": letters[-1][1] = True
        elif "α" <= c <= "ω" or c == "ς": letters.append([c.replace("ς","σ"), False])
    idx = max((i for i, (_, acc) in enumerate(letters) if acc), default=None)
    if idx is None: return None
    t = "".join(b for b, _ in letters[idx:])
    for x, y in {"ει": "ι", "οι": "ι", "αι": "ε", "η": "ι", "υ": "ι", "ω": "ο"}.items(): t = t.replace(x, y)
    return t.replace("σσ", "σ").replace("λλ", "λ").replace("νν", "ν")

def rhyme_ok(a, b):
    if last_word(a) == last_word(b): return False
    if last_word(a) in STOP or last_word(b) in STOP: return False
    if re.search(r"[-¬]$", a) or re.search(r"[-¬]$", b): return False
    ta, tb = accented_tail(a), accented_tail(b)
    if ta is not None and tb is not None:
        return ta == tb and len(last_word(a)) > 2
    ka, kb = rhyme_key(a), rhyme_key(b)
    return len(ka) >= 2 and ka == kb

def greekish(line):
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 15: return False
    g = sum(1 for c in letters if "Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿")
    return g / len(letters) > 0.95

DIAL = re.compile(r"τζ|κκ|ππ|ττ|σσ|θθ|δκ|\bεν\b|\bεννα\b|\bθεννα\b|ουλλ|ουσιν\b|ασιν\b|σιει|σιη|\bποττε|\bαμμα|\bσαν\b|\bμεν\b|\bπου\b")
STOP = set("του της τους των τον την το τα τη και να με σε για απο που ο η οι ως δε θα αν στο στη στην στον στα στις στους μας σας εις κατα μετα προς ενος μιας ενα μια".split())
def dialect_score(t):
    return len(DIAL.findall(strip_acc(t)))

def clean(line):
    line = line.replace("'", "’").replace("`", "’")
    line = re.sub(r"\s+\d{1,4}\s*$", "", line)          # margin numbers
    line = re.sub(r"^\s*\d{1,4}[\.\)]?\s+", "", line)     # leading numbers
    line = re.sub(r"\*", "", line)
    return re.sub(r"\s+", " ", line).strip(" -–—")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--source", required=True); ap.add_argument("--url", default="")
    ap.add_argument("--min", type=int, default=12); ap.add_argument("--max", type=int, default=18)
    ap.add_argument("--min-dialect", type=int, default=0)
    ap.add_argument("--min-run", type=int, default=1, help="keep couplets only inside runs of >=N couplets (verse blocks)")
    ap.add_argument("--run-bypass-dialect", type=int, default=99, help="keep isolated couplets if dialect_score >= this")
    a = ap.parse_args()
    raw = open(a.path, encoding="utf-8", errors="replace").read()
    pages = raw.split("\f")
    out = 0
    for pi, pg in enumerate(pages, 1):
        lines = [clean(l) for l in pg.split("\n")]
        lines = [l for l in lines if l]
        cands = []
        i = 0
        while i < len(lines) - 1:
            l1, l2 = lines[i], lines[i + 1]
            ok = all(greekish(l) and a.min <= syllables(l) <= a.max for l in (l1, l2)) and rhyme_ok(l1, l2) and dialect_score(l1 + " " + l2) >= a.min_dialect
            if ok:
                t = l1 + "\n" + l2
                rec = {"text": t, "source": a.source, "url": a.url, "page": pi, "ocr_confidence": "medium",
                       "syllables": [syllables(l1), syllables(l2)], "dialect_score": dialect_score(t),
                       "notes": "auto-extracted (heuristic: consecutive rhyming 12-18 syllable lines)"}
                cands.append((i, rec)); i += 2
            else:
                i += 1
        # verse-block filter
        runs = []
        for k, (i, rec) in enumerate(cands):
            if runs and i - runs[-1][-1][0] <= 3: runs[-1].append((i, rec))
            else: runs.append([(i, rec)])
        for run in runs:
            for i, rec in run:
                if len(run) >= a.min_run or rec["dialect_score"] >= a.run_bypass_dialect:
                    rec["run_len"] = len(run)
                    print(json.dumps(rec, ensure_ascii=False)); out += 1
    print(out, "couplets", file=sys.stderr)
main()
