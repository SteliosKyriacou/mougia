"""Fetch full info json (description, subtitles list), subtitles and top comments
for relevant catalog entries. No media download.

Writes data/raw/youtube/meta/<id>.info.json, <id>.*.vtt, and a compact
<id>.comments.json (text/author/likes only).

Usage: .venv/bin/python -I scripts/collect/youtube/fetch_meta.py [--limit N] [--max-comments 100]
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/raw/youtube"
META = OUT / "meta"
YTDLP = str(ROOT / ".venv/bin/yt-dlp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10**9)
    ap.add_argument("--max-comments", type=int, default=100)
    a = ap.parse_args()
    META.mkdir(parents=True, exist_ok=True)
    cands = [json.loads(l) for l in open(OUT / "catalog.jsonl")]
    cands = [c for c in cands if c.get("relevant_guess")][: a.limit]
    fails = 0
    unav_p = META / "_unavailable.txt"
    unav = set(unav_p.read_text().split()) if unav_p.exists() else set()
    cands = [c for c in cands if c["id"] not in unav]
    for i, c in enumerate(cands):
        vid = c["id"]
        if (META / f"{vid}.info.json").exists():
            continue
        cmd = [YTDLP, "--js-runtimes", "node", "--skip-download", "--write-info-json",
               "--write-subs", "--write-auto-subs", "--sub-langs", "el-orig", "--sub-format", "vtt",
               "--write-comments", "--extractor-args",
               f"youtube:max_comments={a.max_comments},all,0,0;comment_sort=top",
               "--sleep-requests", "1", "--sleep-subtitles", "6", "--sleep-interval", "2", "--no-progress",
               "-o", str(META / "%(id)s.%(ext)s"), "--", vid]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            if re.search(r"This video is not available|Private video|Video unavailable|confirm your age", r.stderr):
                with open(unav_p, "a") as uf:
                    uf.write(vid + "\n")
            else:
                fails += 1
            print("ERR", vid, r.stderr[-400:], file=sys.stderr)
            if re.search(r"confirm you.re not a bot|HTTP Error 429", r.stderr):
                sys.exit("YouTube bot-check / rate limit; stopping.")
            if fails > 60:
                sys.exit("too many failures; stopping")
            continue
        p = META / f"{vid}.info.json"
        if p.exists():  # shrink: drop formats etc., keep comments separately
            d = json.loads(p.read_text())
            com = [{"text": x.get("text"), "author": x.get("author"), "likes": x.get("like_count"),
                    "id": x.get("id"), "parent": x.get("parent")} for x in (d.get("comments") or [])]
            (META / f"{vid}.comments.json").write_text(json.dumps(com, ensure_ascii=False, indent=0))
            keep = {k: d.get(k) for k in ["id", "title", "description", "channel", "channel_id",
                    "uploader", "upload_date", "duration", "view_count", "like_count", "tags",
                    "categories", "chapters", "webpage_url", "comment_count"]}
            keep["subtitles_langs"] = list((d.get("subtitles") or {}).keys())
            keep["auto_caption_el"] = any(k.startswith("el") for k in (d.get("automatic_captions") or {}))
            p.write_text(json.dumps(keep, ensure_ascii=False, indent=1))
        print(f"[{i+1}/{len(cands)}] {vid} {c['title'][:60]}", file=sys.stderr)


if __name__ == "__main__":
    main()
