"""Shared helpers for web tsiattista scrapers. Run with python3 -I."""
import json, os, re, time, hashlib, urllib.parse
import requests
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (compatible; mougia-research/0.1; academic dialect corpus)"
ROOT = "/home/stelios/repos/mougia/data/raw"
_last = {}
S = requests.Session()
S.headers["User-Agent"] = UA


def sitedir(name):
    d = os.path.join(ROOT, f"web_{name}")
    os.makedirs(os.path.join(d, "html"), exist_ok=True)
    return d


def fname_for(url, ext=".html"):
    p = urllib.parse.urlparse(url)
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", (p.netloc + p.path + ("_" + p.query if p.query else "")))[:120]
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{base}_{h}{ext}"


def get(url, site, ext=".html", delay=1.2, cache=True, **kw):
    """Fetch url politely, save raw to html/, return text (cached)."""
    d = sitedir(site)
    fp = os.path.join(d, "html", fname_for(url, ext))
    if cache and os.path.exists(fp):
        with open(fp, encoding="utf-8") as f:
            return f.read()
    if cache and os.path.exists(fp + ".gz"):
        import gzip
        with gzip.open(fp + ".gz", "rt", encoding="utf-8") as f:
            return f.read()
    host = urllib.parse.urlparse(url).netloc
    wait = delay - (time.time() - _last.get(host, 0))
    if wait > 0:
        time.sleep(wait)
    for attempt in range(3):
        try:
            r = S.get(url, timeout=30, **kw)
            break
        except requests.RequestException as e:
            if attempt == 2:
                raise
            time.sleep(3)
    _last[host] = time.time()
    if r.status_code != 200:
        print("HTTP", r.status_code, url)
        return None
    r.encoding = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else "utf-8"
    t = r.text
    with open(fp, "w", encoding="utf-8") as f:
        f.write(t)
    return t


GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")


def greek_ratio(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0
    return sum(1 for c in letters if GREEK.match(c)) / len(letters)


def html_lines(html_fragment):
    """Strip HTML, return list of non-empty text lines (br / p boundaries)."""
    soup = BeautifulSoup(html_fragment, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for blk in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "blockquote", "tr"]):
        blk.insert_after("\n")
    txt = soup.get_text()
    return [l.strip() for l in txt.replace("\r", "\n").split("\n") if l.strip()]


VOW = "αεηιουωάέήίόύώϊϋΐΰ"


def approx_syll(line):
    """Crude vowel-group count (no synizesis) for heuristics only."""
    s = line.lower()
    s = re.sub(r"[^" + VOW + "a-z ]", "", s)
    # treat common digraphs as one vowel
    s = re.sub(r"(αι|ει|οι|ου|υι|αί|εί|οί|ού|αυ|ευ|αύ|εύ)", "α", s)
    return len(re.findall("[" + VOW + "]", s))


import unicodedata


def _norm_end(line):
    s = unicodedata.normalize("NFD", line.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^α-ωςa-z]", "", s).replace("ς", "σ")
    for a, b in (("ει", "ι"), ("οι", "ι"), ("υι", "ι"), ("αι", "ε"), ("ω", "ο"), ("η", "ι"), ("υ", "ι"), ("ττ", "τ"), ("σσ", "σ"), ("λλ", "λ"), ("νν", "ν"), ("κκ", "κ"), ("ππ", "π"), ("μμ", "μ")):
        s = s.replace(a, b)
    return s


def rhymes(l1, l2):
    a, b = _norm_end(l1), _norm_end(l2)
    return len(a) >= 2 and len(b) >= 2 and a[-2:] == b[-2:]


def is_verse(l, lo=11, hi=21):
    return lo <= approx_syll(l) <= hi and len(l) <= 110


def group_couplets(lines, need_rhyme=False):
    """Group verse lines into couplets. Returns list of (raw_lines, text).
    Long lines (>=11 approx syll) pair 2-by-2; short lines (5-10) group 4-by-4."""
    out = []
    buf = []
    for l in lines:
        buf.append(l)
    i = 0
    while i < len(buf):
        n = approx_syll(buf[i])
        if is_verse(buf[i]) and i + 1 < len(buf) and is_verse(buf[i + 1]) and (not need_rhyme or rhymes(buf[i], buf[i + 1])):
            out.append(buf[i:i + 2]); i += 2
        elif 5 <= n <= 10 and i + 3 < len(buf) and all(5 <= approx_syll(x) <= 10 and len(x) < 60 for x in buf[i:i + 4]) and (not need_rhyme or rhymes(buf[i + 1], buf[i + 3])):
            out.append(buf[i:i + 4]); i += 4
        else:
            i += 1
    return out


def write_jsonl(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(recs), "->", path)


# ---------- better syllable counting & reflow of wrapped verse ----------
_V = set("αεηιουω")


def syllables(text):
    """Approximate Greek syllable count with digraphs and synizesis of unaccented ι/υ before vowel."""
    s = unicodedata.normalize("NFD", text.lower())
    chars = []  # (base, accented, diaeresis)
    for c in s:
        if unicodedata.combining(c):
            if chars:
                b, a, d = chars[-1]
                if c in "́́̀͂":
                    a = True
                elif c == "̈":
                    d = True
                chars[-1] = (b, a, d)
        else:
            chars.append((c, False, False))
    n = 0
    i = 0
    L = len(chars)
    while i < L:
        b, a, d = chars[i]
        if b not in _V:
            i += 1
            continue
        j = i + 1
        if j < L:
            b2, a2, d2 = chars[j]
            if (b + b2) in ("αι", "ει", "οι", "ου", "υι", "αυ", "ευ", "ηυ") and not d2:
                j += 1
        # synizesis: unaccented ι/υ (or ει/οι) followed by a vowel, preceded by consonant
        if j < L and chars[j][0] in _V and not chars[j - 1][1] and chars[j - 1][0] in "ιυ" and not chars[j - 1][2]:
            prev = chars[i - 1][0] if i > 0 else " "
            if prev.isalpha() and prev not in _V:
                i = j
                continue
        n += 1
        i = j
    return n


NAME_RE = re.compile(r"^(?:[Α-ΩΆΈΉΊΌΎΏA-Z][\w΄'’.-]*\.?\s*){1,3}$")


def is_signature(line):
    l = line.strip()
    if len(l) > 30 or l.endswith(",") or syllables(l) > 9:
        return False
    return bool(NAME_RE.match(l)) and len(l.split()) <= 3


def reflow_couplets(lines, lo=12, hi=19):
    """Reflow arbitrarily-wrapped verse lines into 15-syllable couplets via DP.
    Returns list of (verse1, verse2, rhyme_ok)."""
    words, line_end, line_of = [], [], []
    for li, l in enumerate(lines):
        ws = l.split()
        for k, w in enumerate(ws):
            words.append(w)
            line_end.append(k == len(ws) - 1)
            line_of.append(li)
    n = len(words)
    if n == 0:
        return []
    syl = [syllables(w) for w in words]
    pre = [0]
    for x in syl:
        pre.append(pre[-1] + x)
    NEG = -1e9
    best = [NEG] * (n + 1)
    back = [None] * (n + 1)
    best[0] = 0
    for i in range(n):
        if best[i] == NEG:
            continue
        # skip a word
        if best[i] - 1.0 > best[i + 1]:
            best[i + 1] = best[i] - 1.0
            back[i + 1] = (i, None)
        for k in range(i + 1, min(n, i + 14) + 1):
            s1 = pre[k] - pre[i]
            if s1 > hi:
                break
            if s1 < lo:
                continue
            for j in range(k + 1, min(n, k + 14) + 1):
                s2 = pre[j] - pre[k]
                if s2 > hi:
                    break
                if s2 < lo:
                    continue
                v1 = " ".join(words[i:k]); v2 = " ".join(words[k:j])
                r = rhymes_strict(v1, v2)
                sc = (3.0 if r else 0.3) * 1 + (0.5 if line_end[k - 1] else 0) + (0.7 if line_end[j - 1] else 0)
                sc -= 0.15 * (abs(s1 - 15) + abs(s2 - 15))
                sc += 0.1 * (j - i)  # mild preference to cover words
                if best[i] + sc > best[j]:
                    best[j] = best[i] + sc
                    back[j] = (i, (k, r))
    out = []
    j = n
    while j > 0:
        i, info = back[j]
        if info:
            k, r = info
            out.append((" ".join(words[i:k]), " ".join(words[k:j]), r, lines[line_of[i]:line_of[j - 1] + 1]))
        j = i
    return out[::-1]


def segments(lines, keep=lambda l: greek_ratio(l) > 0.7 and syllables(l) <= 21 and len(l) <= 110):
    """Split lines into verse segments. A segment ends at a prose/non-verse line or a signature line.
    Returns list of {"lines": [...], "author": signature-or-None}."""
    segs, cur = [], []
    def _n(x):
        return re.sub(r"[^\w]", "", unicodedata.normalize("NFD", x.lower())).strip()
    expanded = []
    for l in lines:
        # long lines (couplet written on one line, 25-40 syllables) -> split at punctuation so reflow can use them
        if greek_ratio(l) > 0.7 and 22 <= syllables(l) <= 40 and len(l) <= 220:
            parts = [x.strip() for x in re.split(r"(?<=[,.;!·])\s+", l) if x.strip()]
            if all(syllables(x) <= 21 for x in parts):
                expanded += parts
                continue
        expanded.append(l)
    for l in expanded:
        # drop sung refrains that repeat the end of the previous line ("..ειπά να τραουδήσω")
        if cur and len(_n(l)) >= 6 and (_n(cur[-1]).endswith(_n(l)) or _n(l) in _n(cur[-1])):
            continue
        if is_signature(l) and cur:
            segs.append({"lines": cur, "author": l.strip()}); cur = []
        elif keep(l) and syllables(l) >= 3:
            cur.append(l)
        else:
            if cur:
                segs.append({"lines": cur, "author": None}); cur = []
    if cur:
        segs.append({"lines": cur, "author": None})
    return segs


def units_from_lines(lines):
    """-> list of dicts: text, raw_lines(reflowed verses), rhyme_ok, seg, author"""
    out = []
    for si, sg in enumerate(segments(lines)):
        for v1, v2, r, src in reflow_couplets(sg["lines"]):
            out.append({"text": v1 + "\n" + v2, "verses": [v1, v2], "rhyme_ok": r, "seg": si, "cy_score": cy_score(v1 + " " + v2),
                        "author": sg["author"], "seg_lines": src})
    return out


# ---------- stricter rhyme: tail from the last stressed vowel ----------
_ACC = "́́̀͂"


def _tail(line):
    s = unicodedata.normalize("NFD", line.lower())
    s = re.sub(r"[^\w\u0300-\u036f ]", " ", s)
    words = [w for w in s.split() if re.search("[α-ω]", w)]
    if not words:
        return ""
    w = words[-1]
    if not any(c in _ACC for c in w) and len(words) > 1:
        w = words[-2] + w  # enclitic (πονώ τα, χάρη μου)
    base, acc, dia = [], [], []
    for c in w:
        if unicodedata.combining(c):
            if base:
                if c in _ACC: acc[-1] = True
                if c == "\u0308": dia[-1] = True
        else:
            base.append(c); acc.append(False); dia.append(False)
    V = "αεηιουω"
    idx = max((i for i, a in enumerate(acc) if a), default=-1)
    if idx < 0:
        idx = max((i for i, c in enumerate(base) if c in V), default=0)
    bi = idx
    if bi > 0 and base[bi - 1] in V and not dia[bi] and (base[bi - 1] + base[bi]) in ("αι", "ει", "οι", "ου", "υι", "αυ", "ευ"):
        bi -= 1
    t = "".join(base[bi:]).replace("ς", "σ")
    t = t.replace("ου", "U")
    for a, b in (("ει", "ι"), ("οι", "ι"), ("υι", "ι"), ("αι", "ε"), ("ω", "ο"), ("η", "ι"), ("υ", "ι"),
                 ("ττ", "τ"), ("σσ", "σ"), ("λλ", "λ"), ("νν", "ν"), ("κκ", "κ"), ("ππ", "π"), ("μμ", "μ"), ("ρρ", "ρ")):
        t = t.replace(a, b)
    return t.replace("U", "ου")


def _has_acc(line):
    w = unicodedata.normalize("NFD", line.strip().split()[-1]) if line.strip() else ""
    return any(c in _ACC for c in w)


def rhymes_strict(l1, l2):
    if not (_has_acc(l1) and _has_acc(l2)):  # unaccented (informal) text: compare last 3 normalized letters
        a, b = _norm_end(l1), _norm_end(l2)
        if len(a) >= 3 and a[-3:] == b[-3:]:
            return True
    a, b = _tail(l1), _tail(l2)
    if len(a) < 2 or len(b) < 2:
        return False
    if a == b:
        return True
    # tolerate final ν / σ difference (κάτω / κάτων) and one-consonant onset difference
    a2, b2 = a.rstrip("νσ"), b.rstrip("νσ")
    return len(a2) >= 2 and a2 == b2


CY_MARKERS = re.compile(r"(?:^|\s)(?:τζ\S*|τζι|τζαι|τζιαι|τσ[ιί]\S*|κκ\S*|\S*κκ\S*|εν|έν|ένι|εννά|ννα|θκι\S*|σι[ηή]\S*|\S*σι[εέ]\S*|\S+ουσιν|\S+ουσι|\S+ασιν|ποττέ|τωρά|δαμέ|εγιώ|εσού|γιατί|καρκιά\S*|\S*ουν|\S*ειν|πκιά\S*|\S*ττ\S*)(?=\s|$|[,.;!·])")


def cy_score(text):
    """crude count of Cypriot-dialect markers per 10 words"""
    t = text.lower()
    keys = len(re.findall(r"τζ|θκι|κκ|ττ|σι[ηήεέ]|ννα|\bεν\b|\bέν\b|\bεννά\b|ουσιν\b|ασιν\b|\bδαμέ\b|\bεγιώ|\bτζαι\b|\bτζιαι\b|\bτσαι\b|\bπκιά", t))
    words = max(1, len(t.split()))
    return round(10 * keys / words, 2)


# ---------- dialect heuristic ----------
_STRONG = re.compile(r"τζ|θκι|θκια|σι[ηήεέαάοόυύ]|ψι[ηή]|(?<![α-ωά-ώ])(?:εν|έν|ένι|εννά|εννα|ούλλ\w*|ουλλ\w*|δαμέ|δαμαί|εγιώ|εγιώνι|εσού|εσούνι|ποττέ|τωρά|καρκιά\w*|πκιον|πκιόν|κκελλ\w*|ηντα|ήντα|ίντα|ιντα|λαλώ|λαλεί|αππ\w*|ποξω|σσιύλλ\w*|τζαι|τζιαι|τσαι)(?![α-ωά-ώ])|\w{3,}ουσιν(?![α-ωά-ώ])|\w{3,}ασιν(?![α-ωά-ώ])|\w{3,}[ίι]ν(?![α-ωά-ώ])|κκ|ττ")
_WEAK = re.compile(r"σσ|λλ|νν|μμ|ππ")


def dialect_of(text):
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 15:
        return "unknown"
    poly = sum(1 for c in letters if "ἀ" <= c <= "῿")
    if poly / len(letters) > 0.04:
        return "ancient"
    if greek_ratio(text) < 0.6:
        return "unknown"  # latin script (greeklish) or other
    t = text.lower()
    words = max(1, len(t.split()))
    strong = len(_STRONG.findall(t))
    weak = len(_WEAK.findall(t))
    score = (strong + 0.3 * weak) / words
    if strong >= 2 or (strong >= 1 and score >= 0.04):
        return "cypriot"
    if words >= 6:
        return "smg"
    return "unknown"
