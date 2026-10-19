"""Check tsiattista couplets for meter, rhyme and Cypriot dialect.

Usage:
  python3 -I scripts/check_couplets.py duels.jsonl          # synthetic-duel JSONL (one duel per line, "turns": [{"text": ...}])
  python3 -I scripts/check_couplets.py --text "line1\nline2"

For each couplet prints PASS/FAIL with the reasons: syllable counts per line (13–17 accepted, 15 ideal),
whether the two lines rhyme, the Cypriot-dialect score (must be >= 2.5 with no Cretan / Standard-Greek giveaways).
"""
import argparse
import json
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialect import cypriot_score  # noqa: E402
from greek_meter import is_15_syllable, rhyme_key, rhymes, syllables  # noqa: E402

MIN_CY = 2.0
SMG_GIVEAWAYS = [" και ", " είναι ", " θα ", " δεν ", " τώρα ", " κοιτάζω", " κοιτάω", " μιλάω", " θέλεις να "]


def check(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    problems = []
    if len(lines) != 2:
        return False, [f"needs exactly 2 lines, got {len(lines)}"], {}
    syl = [syllables(l)[1] for l in lines]
    for i, l in enumerate(lines):
        if not is_15_syllable(l):
            problems.append(f"line {i + 1} has {syl[i]} syllables (aim for 15)")
    if not rhymes(*lines):
        problems.append(f"lines don't rhyme ({rhyme_key(lines[0])!r} vs {rhyme_key(lines[1])!r})")
    score, cy, non = cypriot_score(text)
    if non:
        problems.append("contains Cretan / non-Cypriot markers")
    if score < MIN_CY:
        problems.append(f"not Cypriot enough (score {score:.1f} < {MIN_CY}); use τζαι, εν, εννά, έσιει, ούλλοι, -ουσιν, final -ν …")
    low = " " + re.sub(r"[^\w’']+", " ", text.lower()) + " "
    smg = [g.strip() for g in SMG_GIVEAWAYS if g in low]
    if smg:
        problems.append("Standard-Greek forms: " + ", ".join(smg))
    return not problems, problems, {"syllables": syl, "cy_score": round(score, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--text")
    a = ap.parse_args()
    if a.text:
        ok, probs, info = check(a.text.replace("\\n", "\n"))
        print("PASS" if ok else "FAIL", info, "; ".join(probs))
        return
    total = passed = 0
    for n, line in enumerate(open(a.path, encoding="utf-8")):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"line {n + 1}: invalid JSON ({e})")
            continue
        for k, t in enumerate(d.get("turns", [])):
            ok, probs, info = check(t.get("text", ""))
            total += 1
            passed += ok
            if not ok:
                print(f"{d.get('duel_id', n)} turn {k}: FAIL {info} — {'; '.join(probs)}\n    {t.get('text', '')!r}")
    print(f"\n{passed}/{total} couplets pass ({passed / max(1, total):.0%})")


if __name__ == "__main__":
    main()
