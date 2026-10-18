"""Heuristic miner for tsiattista-like couplets in Cypriot text files.

Usage: python3 -I mine_couplets.py OUT.jsonl SOURCE_NAME FILE [FILE ...]
  (append mode is NOT used; run once with all inputs, or use --append)

Patterns:
  pair2   : two consecutive lines, each ~13-17 syllables, rhyming endings
  split4  : four lines ~8/7/8/7 where lines 2 and 4 rhyme -> joined into two lines
  oneline : one line of ~27-34 syllables split at punctuation into two rhyming halves
"""
import sys, os, re, json, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import greek_meter as gm

DIALECT = re.compile(r"\b(τζ|τζι|τζαι|τζιαι|εν|έν|ούλ|εννά|ενna|σιη|ψυσι|καρκι|πκιο|πού|τζείν|τζιείν|αρκέψ|ποττέ|ποτζεί)", re.I)
DIALECT_TOK = ["τζ", "σι", "ττ", "κκ", "ππ", "ούλλ", "εννά", "καρκ", "λαλ", "εν ", " έν ", "ψυσι", "μιαν", "τούτ", "τζιαι", "τζαι", "πκ", "θκ", "ρκ"]
LATIN = re.compile(r"[A-Za-z]")


def clean(line):
    line = line.strip().strip('"').strip()
    line = re.sub(r"^\W*\d+[.)]\s*", "", line)  # list numbering
    line = re.sub(r"(ΑπάντησηΔιαγραφή|Διαγραφή|Απάντηση)$", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def greek_ratio(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0
    return sum(1 for c in letters if "Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿") / len(letters)


def meter_fit(rng, target, tol):
    lo, hi = rng
    if lo <= target <= hi:
        return 1.0
    d = min(abs(lo - target), abs(hi - target))
    return max(0.0, 1 - d / (tol + 1)) if d <= tol else 0.0


def rhyme_score(a, b):
    ka, aa = gm.rhyme_key(a)
    kb, ab = gm.rhyme_key(b)
    if not ka or not kb:
        return 0.0, ka, kb
    if gm.last_word_norm(a) == gm.last_word_norm(b):
        return 0.3, ka, kb          # identical word = weak (refrain) rhyme
    if ka == kb and len(ka) >= 2:
        s = 1.0 if (aa and ab) else 0.8
    elif ka == kb:
        s = 0.5                     # single-vowel key (oxytone -ί/-ά): common in prose
    elif len(ka) >= 3 and len(kb) >= 3 and ka[-3:] == kb[-3:]:
        s = 0.6
    elif ka[-2:] == kb[-2:] and len(ka) >= 2:
        s = 0.35
    else:
        s = 0.0
    return s, ka, kb


def dialect_score(text):
    t = " " + text.lower() + " "
    hits = sum(1 for tok in DIALECT_TOK if tok in t)
    return min(1.0, hits / 3)


def ok_line(l, lo=4):
    if not l or LATIN.search(l) or greek_ratio(l) < 0.95:
        return False
    if "http" in l or "@" in l or len(gm.words(l)) < 2:
        return False
    return True


def score_pair(l1, l2, kind):
    if kind == "pair2":
        r1, r2 = gm.syllable_range(l1), gm.syllable_range(l2)
        m = (meter_fit(r1, 15, 2) + meter_fit(r2, 15, 2)) / 2
        if not (13 <= (r1[0] + r1[1]) / 2 <= 17 and 13 <= (r2[0] + r2[1]) / 2 <= 17):
            return None
    else:
        r1, r2 = gm.syllable_range(l1), gm.syllable_range(l2)
        m = (meter_fit(r1, 15, 2) + meter_fit(r2, 15, 2)) / 2
        if m == 0:
            return None
    rs, ka, kb = rhyme_score(l1, l2)
    if rs < 0.35:
        return None
    d = dialect_score(l1 + " " + l2)
    score = 0.45 * rs + 0.35 * m + 0.2 * d
    syl = [round((r1[0] + r1[1]) / 2), round((r2[0] + r2[1]) / 2)]
    return round(score, 3), syl, (ka, kb)


def mine_lines(lines):
    """yield (kind, l1, l2, score, syl, keys, lineno)"""
    L = [clean(x) for x in lines]
    n = len(L)
    used = set()
    i = 0
    while i < n:
        # split4: 8/7/8/7
        if i + 3 < n and all(ok_line(L[i + k]) for k in range(4)):
            rs = [gm.syllable_range(L[i + k]) for k in range(4)]
            if all(r[1] >= 6 and r[0] <= 9 for r in rs):
                a, b = L[i] + " " + L[i + 1], L[i + 2] + " " + L[i + 3]
                rr, _, _ = rhyme_score(L[i + 1], L[i + 3])
                if rr >= 0.6:
                    res = score_pair(a, b, "split4")
                    if res:
                        yield ("split4", a, b) + res + (i,)
                        used.update(range(i, i + 4)); i += 4; continue
        if i + 1 < n and ok_line(L[i]) and ok_line(L[i + 1]):
            res = score_pair(L[i], L[i + 1], "pair2")
            if res:
                yield ("pair2", L[i], L[i + 1]) + res + (i,)
                i += 2; continue
        # oneline: a full couplet on one line
        l = L[i]
        if ok_line(l):
            lo, hi = gm.syllable_range(l)
            if hi >= 26 and lo <= 35:
                best = None
                for m in re.finditer(r"[,;.·!:]\s+|\s[-–]\s", l):
                    a, b = l[:m.start() + 1].strip(" -–"), l[m.end():].strip()
                    ra, rb = gm.syllable_range(a), gm.syllable_range(b)
                    if 13 <= sum(ra) / 2 <= 17 and 13 <= sum(rb) / 2 <= 17:
                        res = score_pair(a, b, "pair2")
                        if res and (best is None or res[0] > best[0][0]):
                            best = (res, a, b)
                if best:
                    res, a, b = best
                    yield ("oneline", a, b) + res + (i,)
        i += 1


def norm_key(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("τζ", "τσ")
    return re.sub(r"[^\w]+", "", t)


def main():
    args = sys.argv[1:]
    append = "--append" in args
    args = [a for a in args if a != "--append"]
    out, source, files = args[0], args[1], args[2:]
    seen = set()
    if append and os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            seen.add(norm_key(json.loads(l)["text"]))
    cnt = 0
    with open(out, "a" if append else "w", encoding="utf-8") as f:
        for path in files:
            with open(path, encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
            cands = list(mine_lines(lines))
            lns = [c[-1] for c in cands]
            for idx, (kind, l1, l2, score, syl, keys, ln) in enumerate(cands):
                near = any(0 < abs(o - ln) <= 6 for o in lns[max(0, idx - 2):idx + 3])
                if near:
                    score = min(1.0, round(score + 0.05, 3))
                text = l1 + "\n" + l2
                k = norm_key(text)
                if k in seen:
                    continue
                seen.add(k)
                f.write(json.dumps({"text": text, "source": source,
                                    "file": os.path.relpath(path, "/home/stelios/repos/mougia"),
                                    "line": ln, "kind": kind, "score": score,
                                    "syllables": syl, "rhyme": list(keys)},
                                   ensure_ascii=False) + "\n")
                cnt += 1
    print(f"{source}: {cnt} new candidates", file=sys.stderr)


if __name__ == "__main__":
    main()
