"""Merge data/old/raw/*/extracted.jsonl into data/old/processed/{couplets,pairs}.jsonl with meter/rhyme checks and dedup."""
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dialect import cypriot_score  # noqa: E402
from greek_meter import dedupe_key, is_15_syllable, rhymes, syllables  # noqa: E402

# Curated collections whose content is Cypriot by provenance (archaic spelling hides modern dialect markers):
# only reject on explicit non-Cypriot markers. Everything else must show positive Cypriot evidence.
TRUSTED_CYPRIOT = {"book_sakellarios1891", "book_schcy_keimena", "book_tterlikkas2002", "book_makrinas2008", "corpus_tsiattista_muse"}
MIN_CY_SCORE = 1.5  # calibrated: ~0.6% of Cretan mantinades pass, ~89% of gold tsiattista pass
NON_CY_CONTEXT = re.compile(r"κρήτ|κρητ|χανι|μαντινάδ|μαντιναδ|ηράκλει|ρέθυμν|σφακ", re.IGNORECASE)

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")


def to_couplet_lines(text):
    """Normalize a record into 2 long lines. 4 short lines (8+7) are joined pairwise; repeated lines are dropped."""
    lines = [l.strip(" \t-–—•*") for l in text.replace("\r", "").split("\n")]
    lines = [l for l in lines if GREEK.search(l)]
    dedup = []
    for l in lines:
        if not dedup or dedupe_key(l) != dedupe_key(dedup[-1]):
            dedup.append(l)
    lines = dedup
    if len(lines) == 4 and all(syllables(l)[1] <= 10 for l in lines):
        lines = [lines[0] + " " + lines[1], lines[2] + " " + lines[3]]
    if len(lines) == 3 and sum(syllables(l)[1] <= 10 for l in lines) == 2:
        # e.g. "long line / short / short"
        short = [i for i, l in enumerate(lines) if syllables(l)[1] <= 10]
        if short == [1, 2]:
            lines = [lines[0], lines[1] + " " + lines[2]]
        elif short == [0, 1]:
            lines = [lines[0] + " " + lines[1], lines[2]]
    return lines


def browser_record(r, by_id):
    thread = r.get("thread_id") or r.get("video_id") or r.get("section") or r.get("url") or r["_dir"]
    order = r.get("post_index", r.get("turn_index", r.get("position", r.get("page"))))
    c = by_id.get(r.get("_cid")) if r.get("_status") == "kept" else None
    return {
        "src": r["_dir"], "n": r["_n"], "thread": str(thread), "order": order if isinstance(order, (int, float)) else None,
        "segment": r.get("segment"), "title": r.get("post_title") or r.get("thread_title") or r.get("title") or r.get("section"),
        "author": r.get("author") or r.get("poet") or r.get("speaker"), "date": r.get("date"), "url": r.get("url"),
        "text": r.get("text") or "", "kind": r.get("source_kind") or r.get("kind"), "category": r.get("category_norm"),
        "rid": r.get("id") or r.get("comment_id"), "reply_to": r.get("reply_to") or r.get("reply_to_turn"),
        "reply_to_user": r.get("reply_to_user"), "duel": bool(r.get("duel")), "start": r.get("start"),
        "status": r.get("_status", "unknown"), "cid": r.get("_cid"), "cy": r.get("_cy"),
        "meter_ok": c["meter_ok"] if c else None, "rhyme_ok": c["rhyme_ok"] if c else None, "syl": c["syllables"] if c else None,
    }


def main():
    records, all_records = [], []
    for path in sorted(glob.glob(os.path.join(ROOT, "raw", "*", "extracted.jsonl"))):
        source_dir = os.path.basename(os.path.dirname(path))
        if source_dir.endswith("NONCYPRIOT"):
            continue
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r["_dir"], r["_n"] = source_dir, n
                all_records.append(r)
                if r.get("kind") == "poem" or not r.get("text"):
                    r["_status"] = "whole_poem" if r.get("kind") == "poem" else "empty"
                elif source_dir == "corpus_mined" and r.get("score", 0) < 0.8:
                    r["_status"] = "low_mining_score"
                # Whisper/auto-caption text of sung duels is too noisy (CER ~0.35) to train on.
                elif r.get("source_kind") in ("whisper", "subtitle"):
                    r["_status"] = "noisy_transcript"
                elif r.get("dialect") in ("smg", "ancient") or r.get("category_norm") in ("modern_poetry", "ancient", "prose_other"):
                    r["_status"] = "not_cypriot"
                else:
                    records.append(r)

    seen, couplets, by_orig = {}, [], {}
    stats = collections.Counter()
    for r in records:
        lines = to_couplet_lines(r["text"])
        stats["records"] += 1
        if len(lines) != 2:
            stats["not_2_lines"] += 1
            r["_status"] = "not_a_couplet"
            continue
        key = dedupe_key(" ".join(lines))
        if len(key) < 25:
            stats["too_short"] += 1
            r["_status"] = "too_short"
            continue
        score, _, strong_non = cypriot_score("\n".join(lines))
        context = " ".join(str(r.get(k) or "") for k in ("category", "post_title", "thread_title", "labels"))
        need = 0 if r["_dir"] in TRUSTED_CYPRIOT else MIN_CY_SCORE
        if NON_CY_CONTEXT.search(context):
            need = max(need, 3)  # Cretan exchanges inside Cypriot sites: keep only clearly Cypriot replies
        if strong_non or score < need:
            stats["not_cypriot"] += 1
            r["_status"], r["_cy"] = "not_cypriot", score
            continue
        meter = [is_15_syllable(l) for l in lines]
        ok_rhyme = rhymes(*lines)
        orig_id = f"{r['_dir']}:{r['_n']}"
        if key in seen:
            stats["duplicate"] += 1
            seen[key]["also_in"].append(r["_dir"])
            by_orig[orig_id] = seen[key]["id"]
            r["_status"], r["_cid"] = "duplicate", seen[key]["id"]
            continue
        c = {
            "id": f"c{len(couplets):06d}",
            "text": "\n".join(lines),
            "lines": lines,
            "syllables": [syllables(l)[1] for l in lines],
            "meter_ok": all(meter),
            "rhyme_ok": ok_rhyme,
            "source": r["_dir"],
            "url": r.get("url"),
            "author": r.get("author") or r.get("poet"),
            "confidence": r.get("confidence") or r.get("ocr_confidence"),
            "category": r.get("category_norm"),
            "cy_score": score,
            "also_in": [],
            "_raw": r,
        }
        seen[key] = c
        by_orig[orig_id] = c["id"]
        r["_status"], r["_cid"], r["_cy"] = "kept", c["id"], score
        couplets.append(c)

    # Reply pairs: (1) explicit reply_to links; (2) consecutive verse turns by different authors in one thread/video.
    by_id = {c["id"]: c for c in couplets}
    thread_of = lambda r: r.get("thread_id") or r.get("video_id") or r.get("url")
    raw_index, threads = {}, collections.defaultdict(list)
    for r in records:
        cid = by_orig.get(f"{r['_dir']}:{r['_n']}")
        if cid is None:
            continue
        for k in ("comment_id", "id", "turn_index"):
            if r.get(k) is not None:
                raw_index[(r["_dir"], thread_of(r), str(r[k]))] = cid
                raw_index[(r["_dir"], None, str(r[k]))] = cid
        if thread_of(r) and (r.get("duel") or r.get("thread_id", "").startswith("topic-") or r.get("video_id")):
            order = (r.get("post_index", r.get("turn_index", r.get("position", 0))) or 0, r.get("segment", 0) or 0, r["_n"])
            threads[(r["_dir"], thread_of(r))].append((order, cid, r.get("author") or r.get("poet") or r.get("speaker")))

    pairs, seen_pairs = [], set()

    def add_pair(call, reply, kind, src):
        if call == reply or (call, reply) in seen_pairs:
            return
        seen_pairs.add((call, reply))
        a, b = by_id[call], by_id[reply]
        pairs.append({"call": call, "reply": reply, "call_text": a["text"], "reply_text": b["text"], "source": src, "link": kind,
                      "good": a["meter_ok"] and a["rhyme_ok"] and b["meter_ok"] and b["rhyme_ok"]})

    for r in records:
        tgt = r.get("reply_to") if r.get("reply_to") is not None else r.get("reply_to_turn")
        cid = by_orig.get(f"{r['_dir']}:{r['_n']}")
        if tgt is None or cid is None:
            continue
        parent = raw_index.get((r["_dir"], thread_of(r), str(tgt))) or raw_index.get((r["_dir"], None, str(tgt)))
        if parent:
            add_pair(parent, cid, "explicit", r["_dir"])
    for (src, _), turns in threads.items():
        turns.sort(key=lambda t: t[0])
        for (_, prev, pa), (_, cur, ca) in zip(turns, turns[1:]):
            if pa and ca and pa != ca:
                add_pair(prev, cur, "consecutive", src)

    os.makedirs(os.path.join(ROOT, "processed"), exist_ok=True)
    with open(os.path.join(ROOT, "processed", "couplets.jsonl"), "w", encoding="utf-8") as f:
        for c in couplets:
            c = {k: v for k, v in c.items() if k != "_raw"}
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(ROOT, "processed", "pairs.jsonl"), "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Every raw record in stored order, with what happened to it — used by the dataset browser (app/).
    with open(os.path.join(ROOT, "processed", "records.jsonl"), "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(browser_record(r, by_id), ensure_ascii=False) + "\n")

    per_src = collections.Counter(c["source"] for c in couplets)
    good = sum(c["meter_ok"] and c["rhyme_ok"] for c in couplets)
    print(json.dumps(dict(stats), ensure_ascii=False))
    print(f'unique couplets: {len(couplets)}  (meter+rhyme ok: {good})  reply pairs: {len(pairs)} (both good: {sum(p["good"] for p in pairs)})')
    for s, n in per_src.most_common():
        g = sum(c["meter_ok"] and c["rhyme_ok"] for c in couplets if c["source"] == s)
        print(f"  {s:40s} {n:6d}  good={g}")


if __name__ == "__main__":
    main()
