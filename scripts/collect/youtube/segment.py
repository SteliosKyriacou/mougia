"""Segment whisper transcripts into performer turns and couplets.

Heuristics (sung tsiattista over violin/lute):
  * whisper segments ~ half-lines or lines; the violin plays ~4-7 s between
    lines of one singer and a longer ritornello (~9-15 s) between singers.
    A gap >= --turn-gap seconds starts a new turn (assumed: new singer).
  * fillers (Ε, Εε, Αι, ρε ...) are stripped; repeated tails ("X, X") and
    immediately repeated segments are collapsed.
  * short fragments are merged until ~12+ syllables to form lines.
  * inside each turn the best consecutive line pair (metre 12-18 + rhyme)
    is the couplet; no rhyming pair -> last two lines, low confidence.
Writes data/old/raw/youtube/extracted_whisper.jsonl and turns/<id>.json.
"""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from greek_verse import score_pair, syllables, strip_acc  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/old/raw/youtube"
TR = OUT / "transcripts"

FILLER = re.compile(r"^(?:(?:ε+|εμ+|μμ+|ελ|α+ι*|αι|ρε|ε ρε|ναι|ωχ|ω+|α|ου+|αμαν)(?:[\s,!.…]+|$))+", re.I)
JUNK = re.compile(r"ευχαριστ|υπότιτλ|authorwave|subtitles|μουσική|χειροκρότ|bravo|μπράβο", re.I)


def norm_words(s):
    return re.findall(r"[^\W\d_]+", strip_acc(s))


def collapse_repeat(text):
    """'A B C, B C' -> 'A B C' (repeated tail of 2+ words)."""
    toks = text.split()
    nw = [strip_acc(re.sub(r"[^\w]", "", t)) for t in toks]
    changed = True
    while changed:
        changed = False
        n = len(nw)
        for k in range(n // 2, 1, -1):
            # does the last k words repeat the k words before them (allow 1 mismatch)?
            a, b = nw[n - 2 * k:n - k], nw[n - k:]
            if len(a) == k and sum(x != y for x, y in zip(a, b)) <= (1 if k >= 4 else 0):
                toks, nw = toks[:n - k], nw[:n - k]
                changed = True
                break
    return " ".join(toks).strip(" ,.;")


def clean_seg(t):
    t = re.sub(r"(\w)\1{3,}", r"\1", t.strip())  # "Εμμμμμμ" -> "Εμ"
    t = FILLER.sub("", t)
    t = re.sub(r"\s+(νης|νη|ρε|δα|να|νω|νο)[.,!]*$", "", t)  # sung tail syllables
    t = collapse_repeat(t)
    return t.strip(" ,.;!…-")


def build_lines(segs):
    lines, buf, bstart, conf = [], "", None, []
    last_key = None
    for s in segs:
        t = clean_seg(s["text"])
        if not t or JUNK.search(t):
            continue
        key = " ".join(norm_words(t))
        if key == last_key:  # whole line repeated
            continue
        last_key = key
        if buf and syllables(buf) < 12:
            buf = buf + " " + t
        else:
            if buf:
                lines.append({"text": collapse_repeat(buf), "start": bstart, "lp": sum(conf) / len(conf)})
            buf, bstart, conf = t, s["start"], []
        conf.append(s["avg_logprob"])
    if buf:
        lines.append({"text": collapse_repeat(buf), "start": bstart, "lp": sum(conf) / len(conf)})
    # dedupe repeated lines inside the turn
    out, seen = [], set()
    for l in lines:
        k = " ".join(norm_words(l["text"]))
        if k in seen:
            continue
        seen.add(k)
        out.append(l)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-gap", type=float, default=8.5)
    ap.add_argument("--src", default="transcripts")
    ap.add_argument("--kind", default="whisper")
    a = ap.parse_args()
    TR = OUT / a.src
    tdir = OUT / ("turns" if a.kind == "whisper" else f"turns_{a.kind}")
    tdir.mkdir(exist_ok=True)
    recs = []
    for p in sorted(TR.glob("*.json")):
        d = json.loads(p.read_text())
        segs = [s for s in d["segments"] if s["text"].strip()]
        turns, cur = [], []
        for s in segs:
            if cur and s["start"] - cur[-1]["end"] >= a.turn_gap:
                turns.append(cur); cur = []
            cur.append(s)
        if cur:
            turns.append(cur)
        T = []
        for ti, tsegs in enumerate(turns):
            lines = build_lines(tsegs)
            if not lines:
                continue
            best = None
            for i in range(len(lines) - 1):
                c, s1, s2, rh = score_pair(lines[i]["text"], lines[i + 1]["text"])
                if best is None or c > best[0]:
                    best = (c, i, rh)
            T.append({"turn_index": len(T), "start": tsegs[0]["start"], "end": tsegs[-1]["end"],
                      "lines": lines, "best_pair": best})
        (tdir / f"{d['id']}.json").write_text(json.dumps(T, ensure_ascii=False, indent=1))
        for t in T:
            if t["best_pair"] is None:
                continue
            c, i, rh = t["best_pair"]
            l1, l2 = t["lines"][i], t["lines"][i + 1]
            asr = (l1["lp"] + l2["lp"]) / 2  # avg logprob, ~ -0.2 good .. -0.8 bad
            asr_c = max(0.0, min(1.0, 1 + asr))  # crude
            conf = round(min(0.5, 0.15 + 0.35 * c) * asr_c, 3)
            recs.append({"text": f"{l1['text']}\n{l2['text']}", "video_id": d["id"], "url": d["url"],
                         "source_kind": a.kind, "turn_index": t["turn_index"], "start": l1["start"],
                         "reply_to_turn": t["turn_index"] - 1 if t["turn_index"] > 0 else None,
                         "rhyme_ok": rh, "n_lines_in_turn": len(t["lines"]),
                         "turn_text": " / ".join(x["text"] for x in t["lines"]), "confidence": conf,
                         **({"subtitle_type": "auto"} if a.kind == "subtitle" else {})})
    with open(OUT / f"extracted_{a.kind}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(recs)} {a.kind} couplets from {len(list(TR.glob('*.json')))} transcripts", file=sys.stderr)


if __name__ == "__main__":
    main()
