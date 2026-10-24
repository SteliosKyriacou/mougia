"""For videos whose top comments already contain verse, refetch comments WITH
reply threads (commenters often answer a couplet with a couplet = real
call->reply pairs). Overwrites meta/<id>.comments.json (superset).
Usage: .venv/bin/python -I scripts/collect/youtube/fetch_replies.py
"""
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/old/raw/youtube"
META = OUT / "meta"
YTDLP = str(ROOT / ".venv/bin/yt-dlp")


def main():
    vids = sorted({json.loads(l)["video_id"] for l in open(OUT / "extracted_text.jsonl")
                   if json.loads(l)["source_kind"] == "comment"})
    for vid in vids:
        flag = META / f"{vid}.replies_done"
        if flag.exists():
            continue
        cmd = [YTDLP, "--js-runtimes", "node", "--skip-download", "--write-comments", "--no-write-info-json",
               "--extractor-args", "youtube:max_comments=300,all,200,30;comment_sort=top",
               "--sleep-requests", "1", "-J", "--", vid]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print("ERR", vid, r.stderr[-300:], file=sys.stderr)
            if re.search(r"confirm you.re not a bot|HTTP Error 429", r.stderr):
                sys.exit("YouTube bot-check / rate limit; stopping.")
            continue
        d = json.loads(r.stdout)
        com = [{"text": x.get("text"), "author": x.get("author"), "likes": x.get("like_count"),
                "id": x.get("id"), "parent": x.get("parent"), "timestamp": x.get("timestamp")}
               for x in (d.get("comments") or [])]
        (META / f"{vid}.comments.json").write_text(json.dumps(com, ensure_ascii=False, indent=0))
        flag.write_text("")
        print(vid, len(com), "comments (with replies)", file=sys.stderr)


if __name__ == "__main__":
    main()
