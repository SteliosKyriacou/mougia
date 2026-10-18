"""Build couplets from visual transcriptions of Sakellarios vol.2 songs section (leaves 108-307).
Input: dir of nNNN.txt files ('# leaf', '## title', verse lines, '> prose', '[fn] ...').
Per song, lines are greedily paired when adjacent lines rhyme (strict or folk-assonance rule of
extract_poem_couplets.rhymes). A song counts as a 'couplet song' if >=50% of its verse lines fall in
rhyming pairs; only those songs' pairs are emitted (ballads in unrhymed political verse are skipped).
Writes songs_text.txt (full clean verse) and extracted_songs.jsonl into the book dir."""
import sys, os, re, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "extract_poem_couplets.py"), encoding="utf-8").read().replace("\nmain()\n", "\n")
g = {"__file__": os.path.join(HERE, "extract_poem_couplets.py")}; exec(compile(src, "epc", "exec"), g)
rhymes, syll = g["rhymes"], g["syll"]
THR = float(os.environ.get("THR", "0.35"))
TX, OUT = sys.argv[1], "/home/stelios/repos/mougia/data/raw/book_sakellarios1891"
URL = "https://archive.org/details/takypriakatoigeg02sakeuoft/page/n{}"
songs, cur, title, page = [], None, "(untitled, continued)", None
files = sorted(glob.glob(os.path.join(TX, "n*.txt")), key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[0]))
for f in files:
    leaf = int(re.findall(r"\d+", os.path.basename(f))[0])
    for raw in open(f, encoding="utf-8"):
        l = raw.rstrip("\n").strip()
        if l.startswith("# leaf"):
            m = re.search(r"page\s+(\d+)", l); page = int(m.group(1)) if m else None; continue
        if l.startswith("## "):
            cur = {"title": l[3:].strip(), "lines": []}; songs.append(cur); continue
        if not l or l.startswith(">") or l.startswith("[fn]"): continue
        if cur is None: cur = {"title": title, "lines": []}; songs.append(cur)
        if re.fullmatch(r"[\.\s…]+", l) or re.search(r"κτλ\.?\s*ὡς\s*ἄνωτ", l): 
            cur["lines"].append(None); continue
        cur["lines"].append((l, leaf, page))
recs, txt = [], []
for s in songs:
    L = s["lines"]; verse = [x for x in L if x]
    txt.append("## " + s["title"]); txt += [x[0] for x in verse]; txt.append("")
    pairs, i = [], 0
    while i < len(L) - 1:
        a, b = L[i], L[i + 1]
        if a and b and rhymes(a[0], b[0]) and syll(a[0]) >= 6 and syll(b[0]) >= 6:
            pairs.append((a, b)); i += 2
        else: i += 1
    strict = sum(1 for a, b in pairs if rhymes(a[0], b[0], lenient=False))
    ratio = 2 * len(pairs) / max(1, len(verse)); sratio = 2 * strict / max(1, len(verse))
    s["ratio"] = ratio; s["sratio"] = sratio
    if sratio < THR: continue
    for a, b in pairs:
        sy = [syll(a[0]), syll(b[0])]
        recs.append({"text": a[0] + "\n" + b[0], "source": "Sakellarios, Τα Κυπριακά, τ. Β΄ (1891) — Δημώδη κυπριακά άσματα",
                     "url": URL.format(a[1]), "page": a[2], "leaf": a[1], "section": "songs: " + s["title"],
                     "ocr_confidence": "high" if "{?}" not in a[0] + b[0] and "[?]" not in a[0] + b[0] else "medium",
                     "syllables": sy, "meter": "15" if min(sy) >= 12 else "short",
                     "notes": f"couplet inside a longer rhymed song (song rhyme ratio strict {sratio:.2f} / lenient {ratio:.2f}); visual transcription of scans"})
open(os.path.join(OUT, "songs_text.txt"), "w", encoding="utf-8").write("\n".join(txt))
with open(os.path.join(OUT, "extracted_songs.jsonl"), "w", encoding="utf-8") as o:
    for r in recs: o.write(json.dumps(r, ensure_ascii=False) + "\n")
for s in songs: print(f"{s['sratio']:.2f} {s['ratio']:.2f} {len([x for x in s['lines'] if x]):4d}  {s['title'][:70]}")
print("songs", len(songs), "couplets", len(recs), "verse lines", sum(len([x for x in s['lines'] if x]) for s in songs))
