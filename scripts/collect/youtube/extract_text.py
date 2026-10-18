"""Extract couplets from descriptions, comments and subtitle files in meta/.

Writes data/raw/youtube/extracted_text.jsonl (merged later into extracted.jsonl).
Usage: .venv/bin/python -I scripts/collect/youtube/extract_text.py
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from greek_verse import score_pair, greek_ratio, strip_acc  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/raw/youtube"
META = OUT / "meta"

CREDIT = re.compile(r"^\S+(\s\S+)?\s*:")  # "Βκιολίν: ...", "Γενική επιμέλεια: ..."
URL = re.compile(r"https?://\S+|www\.\S+|#\S+|@\S+")


def clean(l):
    l = URL.sub("", l)
    l = re.sub(r"^[\s\-–—•*\d.)]+", "", l)  # bullets / numbering
    l = re.sub(r"\s+", " ", l).strip(" \t\"'«»“”")
    return l


def candidate_lines(text):
    out = []
    for raw in re.split(r"[\r\n]+", text or ""):
        for part in re.split(r"\s*(?:/{1,2}|\|)\s*", raw):
            l = clean(part)
            if len(l) >= 10 and greek_ratio(l) > 0.9 and not CREDIT.match(l):
                out.append(l)
            else:
                out.append(None)  # break marker
    return out


def split_long(l):
    """A single line holding both verses: split at a comma/space near the middle."""
    from greek_verse import syllables
    if not (24 <= syllables(l) <= 36):
        return None
    best = None
    for m in re.finditer(r"[,;.·!]\s+|\s+", l):
        a, b = l[:m.start()].strip(" ,;."), l[m.end():].strip()
        if not a or not b:
            continue
        conf, s1, s2, rh = score_pair(a, b)
        if conf >= 0.75 and (best is None or abs(s1 - s2) < best[0]):
            best = (abs(s1 - s2), a, b, conf)
    return best


def merge_half_lines(lines):
    """Lines written as 8+7 half-lines on separate rows -> one 15-syllable line."""
    from greek_verse import syllables
    out, i = [], 0
    while i < len(lines):
        a = lines[i]
        b = lines[i + 1] if i + 1 < len(lines) else None
        if a and b:
            sa, sb = syllables(a), syllables(b)
            if 5 <= sa <= 10 and 5 <= sb <= 10 and 13 <= sa + sb <= 17:
                out.append(a.rstrip(" ,") + ", " + b)
                i += 2
                continue
        out.append(a)
        i += 1
    return out


def pairs_from_text(text):
    lines = merge_half_lines(candidate_lines(text))
    res, i = [], 0
    while i < len(lines):
        l1 = lines[i]
        l2 = lines[i + 1] if i + 1 < len(lines) else None
        if l1 and l2:
            conf, s1, s2, rh = score_pair(l1, l2)
            if conf >= 0.75:
                res.append((l1, l2, conf))
                i += 2
                continue
        if l1:
            b = split_long(l1)
            if b:
                res.append((b[1], b[2], round(b[3] - 0.1, 2)))
        i += 1
    return res


def vtt_text(p):
    out, last = [], None
    for l in p.read_text(errors="replace").splitlines():
        if "-->" in l or l.startswith(("WEBVTT", "Kind:", "Language:")) or not l.strip():
            continue
        l = re.sub(r"<[^>]+>", "", l).strip()
        if l and l != last:
            out.append(l)
            last = l
    return "\n".join(out)


def main():
    recs, seen = [], set()

    def add(l1, l2, conf, vid, kind, extra=None):
        key = (strip_acc(l1 + l2).replace(" ", ""))
        if key in seen:
            return
        seen.add(key)
        r = {"text": f"{l1}\n{l2}", "video_id": vid, "url": f"https://www.youtube.com/watch?v={vid}",
             "source_kind": kind, "confidence": conf}
        if extra:
            r.update(extra)
        recs.append(r)

    for p in sorted(META.glob("*.info.json")):
        d = json.loads(p.read_text())
        vid = d["id"]
        for j, (l1, l2, c) in enumerate(pairs_from_text(d.get("description"))):
            add(l1, l2, c, vid, "description", {"seq_in_text": j})
        cp = META / f"{vid}.comments.json"
        if cp.exists():
            coms = json.loads(cp.read_text())
            has_verse = {}
            for cm in coms:
                ps = pairs_from_text(cm.get("text"))
                if ps:
                    has_verse[cm.get("id")] = True
                cm["_pairs"] = ps
            for cm in coms:
                par = cm.get("parent")
                rt = par if par and par != "root" and has_verse.get(par) else None
                for j, (l1, l2, c) in enumerate(cm["_pairs"]):
                    add(l1, l2, c, vid, "comment", {"comment_id": cm.get("id"), "seq_in_text": j,
                                                    "reply_to_comment": rt})
        for sp in META.glob(f"{vid}.*.vtt"):
            kind = "subtitle"
            # auto-generated captions are ASR; mark separately
            auto = "auto" if not d.get("subtitles_langs") or sp.name.split(".")[1] not in d["subtitles_langs"] else "manual"
            if auto == "auto":
                continue  # auto-captions: arbitrary line breaks -> pairing is noise; kept raw in meta/
            for l1, l2, c in pairs_from_text(vtt_text(sp)):
                add(l1, l2, round(c * (0.6 if auto == "auto" else 1), 2), vid, kind, {"subtitle_type": auto})
    with open(OUT / "extracted_text.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(recs), Counter(r["source_kind"] for r in recs), file=sys.stderr)


if __name__ == "__main__":
    main()
