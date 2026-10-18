"""Syllable-level tokenizer for Cypriot Greek verse.

Each token is one orthographic syllable; a word-initial syllable carries the prefix "▁" (so word boundaries survive),
punctuation is its own token, and <nl> marks a line break inside a couplet. Syllables outside the vocabulary fall back
to single characters ("▁" + chars for word-initial ones), so every text round-trips.

Why syllables: tsiattista are defined by 15-syllable lines and rhyme on the final syllables, so meter becomes
"count to 15" and rhyme becomes "match the last 2–3 tokens".
"""
import collections
import json
import re
import sys
import unicodedata
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from greek_meter import nuclei  # noqa: E402

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<nl>", "<sep>"]
PAD, UNK, BOS, EOS, NL, SEP = range(len(SPECIALS))
WB = "▁"
APOS = "’"
KEEP_CHAR = re.compile(r"[\u0370-\u03ffa-z0-9.,;:!?·«»\"()\-–—’…/]")
TOKEN_RE = re.compile(r"[’']?[^\W\d_]+(?:[’'][^\W\d_]*)*|\d+|[^\s\w]", re.UNICODE)


def normalize(text):
    """NFC, lowercase, polytonic → monotonic (keep the acute as stress), unify apostrophes."""
    out = []
    for ch in unicodedata.normalize("NFD", text.lower()):
        if ch in "̯̓̔ͅ":  # breathings, iota subscript, non-syllabic mark
            continue
        if ch in "̀͂":  # grave / circumflex → acute
            ch = "́"
        out.append(ch)
    t = unicodedata.normalize("NFC", "".join(out))
    t = re.sub(r"[ʼ'`΄´‘]", APOS, t)
    return t.replace("ς", "σ")  # final sigma is restored in decode()


FINAL_SIGMA = re.compile(r"σ(?![^\W\d_])")


def _final_sigma(text):
    return FINAL_SIGMA.sub("ς", text)


def syllabify(word):
    """Orthographic syllables: consonants between nuclei go to the following syllable; trailing consonants stay."""
    base, ns = nuclei(word)
    if len(ns) <= 1:
        return [word]
    # Map positions in the accent-stripped base back to the NFC word (same length after NFD-stripping marks).
    if len(unicodedata.normalize("NFC", word)) != len(base):  # unexpected composition; don't split
        return [word]
    # Split each consonant cluster between two nuclei: geminates (ττ, λλ) and liquid/nasal + consonant (ρκ, λτ, νθ)
    # split after the first consonant; any other cluster (στ, θκ, τσ, μπ, ντ, τζ) is the onset of the next syllable.
    adj = []
    for (_, end, _), (start, _, _) in zip(ns, ns[1:]):
        cl = base[end:start]
        if len(cl) >= 2 and cl[0].isalpha() and (cl[0] == cl[1] or (cl[0] in "ρλμν" and cl[:2] != "μν")):
            adj.append(end + 1)
        else:
            adj.append(end)
    pieces, last = [], 0
    for a in adj:
        pieces.append(word[last:a])
        last = a
    pieces.append(word[last:])
    return [p for p in pieces if p]


def pretokenize(text):
    """Text → list of string tokens (before vocab lookup)."""
    toks = []
    for li, line in enumerate(normalize(text).split("\n")):
        if li:
            toks.append("<nl>")
        for w in TOKEN_RE.findall(line):
            if w[0].isalpha() or (w[0] == APOS and len(w) > 1):
                sy = syllabify(w)
                toks.append(WB + sy[0])
                toks.extend(sy[1:])
            else:
                toks.append(w)
    return toks


class Tokenizer:
    def __init__(self, vocab):
        self.itos = list(vocab)
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @classmethod
    def build(cls, texts, min_count=3, max_size=12000):
        cnt = collections.Counter()
        chars = collections.Counter()
        for t in texts:
            for tok in pretokenize(t):
                cnt[tok] += 1
                for ch in tok.lstrip(WB):
                    chars[ch] += 1
        vocab = list(SPECIALS)
        # character fallback (plain and word-initial)
        for ch, _ in chars.most_common():
            if not KEEP_CHAR.match(ch):
                continue  # emoji, CJK, Cyrillic... map to <unk>
            for v in (ch, WB + ch):
                if v not in vocab:
                    vocab.append(v)
        for tok, n in cnt.most_common():
            if n < min_count or len(vocab) >= max_size:
                break
            if tok not in vocab and tok not in SPECIALS and all(KEEP_CHAR.match(ch) and not ("a" <= ch <= "z") for ch in tok.lstrip(WB)):
                vocab.append(tok)
        return cls(vocab)

    def __len__(self):
        return len(self.itos)

    def encode(self, text, bos=False, eos=False):
        ids = [BOS] if bos else []
        for tok in pretokenize(text):
            if tok in self.stoi:
                ids.append(self.stoi[tok])
            elif tok.startswith(WB):
                ids.append(self.stoi.get(WB + tok[1], UNK))
                ids.extend(self.stoi.get(ch, UNK) for ch in tok[2:])
            else:
                ids.extend(self.stoi.get(ch, UNK) for ch in tok)
        if eos:
            ids.append(EOS)
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            s = self.itos[i] if 0 <= i < len(self.itos) else "<unk>"
            if s in ("<pad>", "<bos>", "<eos>", "<sep>"):
                continue
            if s == "<nl>":
                out.append("\n")
            elif s.startswith(WB):
                out.append(" " + s[1:])
            elif len(s) == 1 and not s.isalpha() and s != APOS:
                out.append(s)  # punctuation hugs the previous word
            else:
                out.append(s)
        text = "".join(out)
        text = re.sub(r" *\n *", "\n", text).strip()
        return _final_sigma(text)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "syllable", "vocab": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["vocab"])
