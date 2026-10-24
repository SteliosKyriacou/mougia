"""Discover tsiattista videos on YouTube via yt-dlp flat searches.

Writes data/old/raw/youtube/search_raw/<query-slug>.json (flat -J output) and
data/old/raw/youtube/catalog.jsonl (deduplicated, with relevance flag).

Usage: .venv/bin/python -I scripts/collect/youtube/discover.py [--n 200]
"""
import argparse, hashlib, json, re, subprocess, sys, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/old/raw/youtube"
YTDLP = str(ROOT / ".venv/bin/yt-dlp")

QUERIES = [
    "τσιαττιστά", "τσιατιστά", "τσιαττιστά φεστιβάλ", "ποιητάρηδες τσιαττιστά",
    "tsiattista", "chattista", "tsiatista", "Κυπριακά τσιατιστά γάμος",
    "τσιαττιστά ΡΙΚ", "τσιατιστά ΡΙΚ αρχείο", "Πάμπος Κουζάλης", "Πάμπος Κουζάλης τσιαττιστά",
    "Γιάννης Αγαπίου τσιαττιστά", "Χαμπής τσιαττιστά", "Πιερής Κυπριανού τσιαττιστά",
    "Τάκης Ρουσιάς τσιαττιστά", "ποιητάρης τσιαττιστά", "τσιαττιστά Λεμεσός",
    "τσιαττιστά αντάμωμα", "τσιαττιστά βιολί λαούτο", "τσιαττιστά μονομαχία",
    "Κυπριακά τσιαττιστά", "τσιαττιστά Κύπρος", "τσιατιστά Κυπριακά",
    "Cyprus Music Archive τσιαττιστά", "τσιαττιστά 2023", "τσιαττιστά 2024",
    "τσιαττιστά διαγωνισμός", "Φεστιβάλ Τσιαττιστών", "τσιαττιστά Αθηένου",
    "chattista cyprus", "tsiattista poetry duel", "Κυπριακή ποιητική μονομαχία",
    "τσιαττιστά Κληρίδης", "Ζαννέττος τσιαττιστά", "Μιχάλης Χαμπής τσιαττιστά",
    "τσιατιστά ποιητάρηδες γάμος", "Κυπριακός γάμος τσιαττιστά",
]

def strip_acc(s):
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()

STRONG = re.compile(r"τσια?τ?τισ[τμ]|παλι[ωο]μα|paliomat|τσατιστ|tsia?tt?ist|chatt?ist|ciatt?ist|τσιαττιστ|ποιηταρ|poiitar|ποιητικη μονομαχ|ποιητικ\w* αναμετρ|αντ[αα]μωμα ποιητ")
NEG = re.compile(r"\b(remix|karaoke|tiktok|cover)\b|τσια?τ?τιστα[ςσ]\b|καστορι|kastori|μπουρμπουτσ")

def slug(q):
    return hashlib.md5(q.encode()).hexdigest()[:10]

# channels that turned out to host many duels (from the search results)
CHANNELS = [
    "https://www.youtube.com/channel/{}/videos",
]
CHANNEL_NAMES = ["Panicos Apostolou", "Μουχταρης 1", "Elias Xm", "nikos katsiantonis",
                 "Medical Gazette", "Αι Νικολιτης", "kyrichr", "Marina Georgiou", "Ρίζες της Κύπρου",
                 "Κυπριακή Παράδοση", "Nikolas Nikolaou", "kourostatis", "Pantelis Hoplaros"]


def run_search(q, n):
    p = OUT / "search_raw" / f"{slug(q)}.json"
    if p.exists():
        return json.loads(p.read_text())
    target = q if q.startswith("http") else f"ytsearch{n}:{q}"
    cmd = [YTDLP, "--js-runtimes", "node", "--flat-playlist", "-J", "--sleep-requests", "1", target]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print("ERR", q, r.stderr[-800:], file=sys.stderr)
        if re.search(r"confirm you.re not a bot|HTTP Error 429", r.stderr):
            sys.exit("YouTube bot-check / rate limit; stopping.")
        return None
    d = json.loads(r.stdout)
    d["_query"] = q
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False))
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    cat = {}
    # pass 1: searches; pass 2: full uploads of channels that dominate relevant hits
    chan_ids = {}
    for f in (OUT / "search_raw").glob("*.json"):
        for e in json.loads(f.read_text()).get("entries") or []:
            if e.get("channel") in CHANNEL_NAMES and e.get("channel_id"):
                chan_ids[e["channel"]] = e["channel_id"]
    chan_urls = [CHANNELS[0].format(c) for c in sorted(chan_ids.values())]
    for q in QUERIES + a.extra + chan_urls:
        d = run_search(q, a.n)
        if not d:
            continue
        ents = d.get("entries") or []
        print(f"{len(ents):4d}  {q}", file=sys.stderr)
        for e in ents:
            vid = e.get("id")
            if not vid or e.get("ie_key", "Youtube") != "Youtube":
                continue
            rec = cat.setdefault(vid, {
                "id": vid, "url": f"https://www.youtube.com/watch?v={vid}",
                "title": e.get("title"), "channel": e.get("channel") or e.get("uploader"),
                "channel_id": e.get("channel_id"), "duration": e.get("duration"),
                "view_count": e.get("view_count"), "desc_snippet": e.get("description"), "queries": []})
            rec["queries"].append(q)
    for rec in cat.values():
        t = strip_acc((rec["title"] or "") + " " + (rec.get("desc_snippet") or ""))
        rec["title_match"] = bool(STRONG.search(t))
        rec["relevant_guess"] = rec["title_match"] and not NEG.search(t) and (rec["duration"] or 0) >= 20
    recs = sorted(cat.values(), key=lambda r: (-r["relevant_guess"], -(r["view_count"] or 0)))
    with open(OUT / "catalog.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"catalogued {len(recs)}, title-relevant {sum(r['relevant_guess'] for r in recs)}", file=sys.stderr)

if __name__ == "__main__":
    main()
