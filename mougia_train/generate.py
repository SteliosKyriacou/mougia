"""Reply generation for the fly-brain poets: raw sampling, and verse mode (meter-constrained + best-of-n by rhyme)."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from dialect import cypriot_score  # noqa: E402
from greek_meter import is_15_syllable, rhymes, syllables  # noqa: E402
from mougia_train.tokenizer import BOS, CONT, DUEL, PAD, SEP, UNK  # noqa: E402

BAN = (PAD, BOS, SEP, UNK, DUEL, CONT)


def score_reply(text):
    lines = [l for l in text.split("\n") if l.strip()]
    two = len(lines) == 2
    return {"syllables": [syllables(l)[1] for l in lines[:2]], "meter_ok": two and all(is_15_syllable(l) for l in lines),
            "rhyme_ok": two and rhymes(*lines), "cy_score": round(cypriot_score(text)[0], 1)}


def verse_masks(tok):
    V = len(tok)
    junk = [i for i, s in enumerate(tok.itos) if any(ch.isdigit() or "a" <= ch <= "z" for ch in s.lstrip("▁"))]
    return {"syllable": torch.tensor([tok.is_syllable(i) for i in range(V)]),
            "word_start": torch.tensor([tok.is_word_start(i) for i in range(V)]), "min_syl": 13, "max_syl": 16,
            "ban": junk}  # digits / Latin letters never belong in a tsiattisto


def prompt_ids(tok, text, style="duel"):
    return tok.encode(text, bos=True) + [SEP, DUEL if style == "duel" else CONT]


def rank(cand):
    """Rhyme matters most, then meter, then sounding Cypriot, then the model's own confidence."""
    s = cand["scores"]
    return 3.0 * s["rhyme_ok"] + 2.0 * s["meter_ok"] + 1.0 * (s["cy_score"] >= 1.5) + 0.2 * cand["logp"]


def reply(model, tok, text, mode="verse", n=16, temperature=0.8, style="duel", masks=None):
    """Returns dict(reply, scores, candidates). mode='raw' samples once with no constraints."""
    ids = prompt_ids(tok, text, style)
    if mode == "raw":
        seqs, lps = model.generate_batch(ids, n=1, temperature=temperature, ban_ids=BAN)
    else:
        masks = masks or verse_masks(tok)
        seqs, lps = model.generate_batch(ids, n=n, temperature=temperature, ban_ids=BAN + tuple(masks["ban"]), verse=masks)
    cands = []
    for s, lp in zip(seqs, lps):
        t = tok.decode(s)
        cands.append({"reply": t, "logp": round(lp, 3), "scores": score_reply(t)})
    cands.sort(key=rank, reverse=True)
    return {"reply": cands[0]["reply"], "scores": cands[0]["scores"], "logp": cands[0]["logp"], "candidates": cands}
