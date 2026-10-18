"""Small helpers for Greek/Cypriot verse: syllable estimate and loose rhyme test."""
import re, unicodedata

GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
VOW = "αεηιουω"
# digraphs counted as one nucleus (no diaeresis)
DIGRAPHS = ["αι", "ει", "οι", "ου", "υι", "αυ", "ευ", "ηυ"]


def strip_acc(s):
    s = unicodedata.normalize("NFD", s.lower())
    keep = []
    for c in s:
        if unicodedata.category(c) == "Mn":
            if c == "̈":  # diaeresis: mark break
                keep.append("¨")
            continue
        keep.append(c)
    return unicodedata.normalize("NFC", "".join(keep)).replace("ς", "σ")


def syllables(line, synizesis=True):
    """Rough syllable count of a Greek line. With synizesis, unaccented ι/υ/ει
    before another vowel merges (κκαρκιά -> κκαρ-κιά), and vowel-vowel across
    word boundary is elided once (very rough, approximates sung metre)."""
    raw = unicodedata.normalize("NFD", line.lower())
    # build list of (base_char, accented, diaeresis)
    chars = []
    for c in raw:
        if unicodedata.category(c) == "Mn":
            if chars:
                if c in "́̀͂":
                    chars[-1][1] = True
                elif c == "̈":
                    chars[-1][2] = True
            continue
        chars.append([c, False, False])
    words, cur = [], []
    for ch in chars:
        if ch[0] in VOW or ch[0].isalpha():
            cur.append(ch)
        else:
            if cur:
                words.append(cur); cur = []
    if cur:
        words.append(cur)
    total = 0
    prev_end_vowel = False
    for w in words:
        n = 0
        i = 0
        nuclei = []
        while i < len(w):
            c, acc, dia = w[i]
            if c in VOW:
                j = i + 1
                if j < len(w) and (c + w[j][0]) in DIGRAPHS and not w[j][2]:
                    j += 1
                # synizesis: unaccented ι/υ (or ει/οι digraph unaccented) followed by vowel
                if synizesis and j < len(w) and w[j][0] in VOW and not w[j][2]:
                    seg = w[i:j]
                    if all(not s[1] for s in seg) and seg[-1][0] in "ιυ":
                        i = j
                        continue
                nuclei.append(i)
                i = j
            else:
                i += 1
        n = len(nuclei)
        if n == 0:
            continue
        starts_vowel = w[0][0] in VOW
        if synizesis and prev_end_vowel and starts_vowel and n >= 1:
            n -= 1  # elision across word boundary (approx.)
        total += n
        prev_end_vowel = w[-1][0] in VOW
    return total


_RHN = [("ει", "ι"), ("οι", "ι"), ("υι", "ι"), ("η", "ι"), ("υ", "ι"), ("ω", "ο"), ("αι", "ε"),
        ("ς", "σ")]


def rhyme_key(line, n=3):
    ws = re.findall(r"[^\W\d_]+", strip_acc(line))
    if not ws:
        return ""
    w = ws[-1]
    for a, b in _RHN:
        w = w.replace(a, b)
    return w[-n:]


def _last_word(line):
    toks = re.findall(r"\w+", strip_acc(line))
    return toks[-1] if toks else ""


def rhymes(l1, l2):
    w1, w2 = _last_word(l1), _last_word(l2)
    if not w1.isalpha() or not w2.isalpha() or w1 == w2 or len(w1) < 2 or len(w2) < 2:
        return False  # ends in a number / identical word / single letter: not a real rhyme
    a, b = rhyme_key(l1, 3), rhyme_key(l2, 3)
    if not a or not b:
        return False
    return a == b or (a[-2:] == b[-2:] and len(a) >= 2)


def greek_ratio(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(GREEK.match(c)) for c in letters) / len(letters)


def score_pair(l1, l2):
    s1, s2 = syllables(l1), syllables(l2)
    ok_m = 12 <= s1 <= 18 and 12 <= s2 <= 18
    good_m = 13 <= s1 <= 17 and 13 <= s2 <= 17
    rh = rhymes(l1, l2)
    if good_m and rh:
        conf = 0.9
    elif ok_m and rh:
        conf = 0.75
    elif good_m:
        conf = 0.5
    else:
        conf = 0.0
    return conf, s1, s2, rh
