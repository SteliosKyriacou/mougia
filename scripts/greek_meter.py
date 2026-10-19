"""Approximate syllable counting, rhyme detection and normalization for Cypriot Greek verse."""
import re
import unicodedata

VOWELS = set("αεηιουω")
# Digraphs that form a single vowel nucleus (unless the second letter carries a diaeresis).
DIGRAPHS = ("αι", "ει", "οι", "υι", "ου", "αυ", "ευ", "ηυ")
FILLERS = re.compile(r"\b(ε{2,}|α{2,}|αμάν|αμαν|γιάλα|γιαλα)\b")


def strip_marks(s):
    """Lowercase, decompose, and return (base_text, accent_flags, diaeresis_flags)."""
    base, acc, dia = [], [], []
    for ch in unicodedata.normalize("NFD", s.lower()):
        if unicodedata.combining(ch):
            if not base:
                continue
            if ch in ("̈",):  # diaeresis
                dia[-1] = True
            elif ch in ("́", "̀", "͂"):  # acute, grave, perispomeni
                acc[-1] = True
            continue
        base.append("σ" if ch == "ς" else ch)
        acc.append(False)
        dia.append(False)
    return "".join(base), acc, dia


def nuclei(word):
    """Return list of (start, end, accented) vowel nuclei for one word, with synizesis of unstressed ι/ε + vowel."""
    b, acc, dia = strip_marks(word)
    out, i = [], 0
    while i < len(b):
        if b[i] not in VOWELS:
            i += 1
            continue
        j = i + 1
        if j < len(b) and b[i:j + 1] in DIGRAPHS and not dia[j]:
            j += 1
        out.append([i, j, any(acc[i:j])])
        i = j
    # synizesis: unaccented ι (or ει/οι/υ) directly followed by another nucleus -> merge (καρκιά, πόθκια)
    merged = []
    for n in out:
        if merged:
            p = merged[-1]
            if p[1] == n[0] and b[p[0]:p[1]] in ("ι", "υ", "ει", "οι", "η") and not p[2] and p[0] > 0 and b[p[0] - 1] not in VOWELS:
                merged[-1] = [p[0], n[1], n[2]]
                continue
        merged.append(n)
    return b, merged


WORD_RE = re.compile(r"[^\W\d_]+(?:['’ʼ][^\W\d_]+)*", re.UNICODE)


def clean_line(line):
    line = unicodedata.normalize("NFC", line)
    line = FILLERS.sub(" ", line.lower())
    return re.sub(r"\s+", " ", line).strip()


def syllables(line):
    """Return (count_without_synaloepha, count_with_synaloepha)."""
    words = WORD_RE.findall(clean_line(line))
    total, merges, prev_vowel_end = 0, 0, False
    for w in words:
        b, ns = nuclei(w)
        if not ns:
            continue
        total += len(ns)
        if prev_vowel_end and b[0] in VOWELS:
            merges += 1
        prev_vowel_end = b[-1] in VOWELS
    return total, total - merges


def rhyme_key(line):
    """Ending from the last stressed vowel (or the penultimate nucleus if unaccented), phonetically normalized."""
    words = WORD_RE.findall(clean_line(line))
    if not words:
        return ""
    b, ns = nuclei(words[-1])
    if not ns:
        return ""
    stressed = [n for n in ns if n[2]]
    start = (stressed[-1] if stressed else ns[-2] if len(ns) > 1 else ns[-1])[0]
    return phonetic(b[start:])


def phonetic(s):
    s = strip_marks(s)[0]
    for a, z in (("ει", "ι"), ("οι", "ι"), ("υι", "ι"), ("αι", "ε"), ("ου", "u"), ("η", "ι"), ("υ", "ι"), ("ω", "ο"),
                 ("τζ", "τσ"), ("σσ", "σ"), ("ττ", "τ"), ("λλ", "λ"), ("ννν", "ν"), ("νν", "ν"), ("κκ", "κ"), ("ππ", "π"), ("μμ", "μ")):
        s = s.replace(a, z)
    return s


def rhymes(a, b):
    ka, kb = rhyme_key(a), rhyme_key(b)
    if not ka or not kb:
        return False
    # a final -ν / -ς doesn't break a rhyme in Cypriot verse (επόνουν / γειτόνου)
    ka, kb = re.sub(r"[νσ]$", "", ka), re.sub(r"[νσ]$", "", kb)
    if ka == kb:
        return True
    # tolerate a differing onset consonant: compare from the stressed vowel onward
    va = re.sub(r"^[^αεηιουωu]+", "", ka)
    vb = re.sub(r"^[^αεηιουωu]+", "", kb)
    return len(va) >= 2 and va == vb


def dedupe_key(text):
    t = phonetic(clean_line(text))
    return re.sub(r"[^α-ωu]", "", t)


def is_15_syllable(line, lo=13, hi=17):
    a, b = syllables(line)
    return lo <= b <= hi or lo <= a <= hi
