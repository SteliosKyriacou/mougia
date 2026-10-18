"""Heuristic Greek syllable counting and rhyme keys (for Cypriot tsiattista mining).

Syllables: vowel nuclei after merging digraphs (αι ει οι ου υι ηυ; αυ/ευ count as
one nucleus too). Unstressed ι/υ/ει/οι between a consonant and a following vowel is
treated as a glide (synizesis: ψυσιή = ψυ-σιή, καρκιά = καρ-κιά). Vowel-vowel word
boundaries can merge (synaloepha), so we return a range [min, max].
"""
import re
import unicodedata

VOWELS = set("αεηιουω")
ACCENT = "́"      # combining acute (tonos) after NFD
DIAER = "̈"
DIGRAPHS = ("αι", "ει", "οι", "ου", "υι", "αυ", "ευ", "ηυ")
GLIDE_NUCLEI = ("ι", "υ", "ει", "οι")


def _letters(word):
    """Return list of (base_letter, stressed, diaeresis) for Greek letters in word."""
    out = []
    for ch in unicodedata.normalize("NFD", word.lower()):
        if ch == ACCENT or ch == "͂" or ch == "̀":
            if out:
                b, _, d = out[-1]
                out[-1] = (b, True, d)
        elif ch == DIAER:
            if out:
                b, s, _ = out[-1]
                out[-1] = (b, s, True)
        elif unicodedata.combining(ch):
            continue
        elif "α" <= ch <= "ω" or ch == "ς":
            out.append(("σ" if ch == "ς" else ch, False, False))
    return out


def nuclei(word):
    """Split a word into vowel nuclei: list of dicts {text, start, end, stressed}."""
    L = _letters(word)
    res, i = [], 0
    while i < len(L):
        b, s, d = L[i]
        if b not in VOWELS:
            i += 1
            continue
        j = i + 1
        txt = b
        stressed = s
        if j < len(L):
            b2, s2, d2 = L[j]
            if (b + b2) in DIGRAPHS and not d2 and not s:
                txt += b2
                stressed = stressed or s2
                j += 1
        res.append(dict(text=txt, start=i, end=j, stressed=stressed))
        i = j
        # hiatus vowels after this nucleus start a new nucleus on next loop
    # synizesis: unstressed glide nucleus preceded by consonant, followed directly by vowel
    merged = []
    k = 0
    while k < len(res):
        n = res[k]
        if (k + 1 < len(res) and n["text"] in GLIDE_NUCLEI and not n["stressed"]
                and n["end"] == res[k + 1]["start"] and n["start"] > 0
                and L[n["start"] - 1][0] not in VOWELS
                and not L[n["start"]][2]):
            nxt = dict(res[k + 1])
            nxt["glide"] = nxt["start"] - n["start"]
            nxt["start"] = n["start"]
            merged.append(nxt)
            k += 2
            continue
        merged.append(n)
        k += 1
    return merged, L


WORD_RE = re.compile(r"[Ͱ-Ͽἀ-῿]+(?:['’΄`][Ͱ-Ͽἀ-῿]+)*")


def words(line):
    return WORD_RE.findall(line)


def syllable_range(line):
    ws = words(line)
    total, syn = 0, 0
    prev_ends_vowel = False
    for w in ws:
        w = re.sub(r"['’΄`]", "", w)
        nu, L = nuclei(w)
        if not nu:
            continue
        total += len(nu)
        starts_vowel = L and L[0][0] in VOWELS
        if prev_ends_vowel and starts_vowel:
            syn += 1
        prev_ends_vowel = L and L[-1][0] in VOWELS
    return max(total - syn, 0), total


_NORM = [("ει", "ι"), ("οι", "ι"), ("υι", "ι"), ("αι", "ε"), ("ου", "U"),
         ("η", "ι"), ("υ", "ι"), ("ω", "ο"), ("U", "ου")]


def _phon(s):
    for a, b in _NORM:
        s = s.replace(a, b)
    s = re.sub(r"(.)\1", r"\1", s)        # double consonants
    s = s.replace("ν", "") if s.endswith("ν") else s  # Cypriot final -ν is unstable
    return s


ENCLITICS = {"μου", "σου", "του", "της", "μας", "σας", "τους", "τον", "την", "το",
             "τα", "τες", "τις", "με", "σε", "μιαν", "τζιαι", "τζαι"}


def rhyme_key(line):
    """Accent-stripped, phonetically-normalised ending from last stressed vowel.
    Returns (key, has_accent)."""
    ws = [re.sub(r"['’΄`]", "", w) for w in words(line)]
    ws = [w for w in ws if nuclei(w)[0]]
    if not ws:
        return None, False
    w = ws[-1]
    nu, L = nuclei(w)
    enclitic = ""
    if (len(ws) > 1 and len(nu) == 1 and not nu[0]["stressed"]
            and last_word_norm(w) in ENCLITICS):
        enclitic = "".join(b for b, _, _ in L)
        w = ws[-2]
        nu, L = nuclei(w)
    idx = None
    for k in range(len(nu) - 1, -1, -1):
        if nu[k]["stressed"]:
            idx = k
            break
    has_acc = idx is not None
    if idx is None:
        idx = max(len(nu) - 2, 0) if len(nu) > 1 else 0
        # an unaccented last word of >1 syllable: guess paroxytone
        if len(nu) == 1 and len(ws) > 1:
            pass
    start = nu[idx]["start"] + nu[idx].get("glide", 0)
    tail = "".join(b for b, _, _ in L[start:]) + enclitic
    return _phon(tail), has_acc


def last_word_norm(line):
    ws = words(line)
    if not ws:
        return ""
    w = unicodedata.normalize("NFD", ws[-1].lower())
    return "".join(c for c in w if not unicodedata.combining(c))
