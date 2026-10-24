"""Convert YouTube auto-caption VTT (meta/<id>.el-orig.vtt) into the transcript
format used by segment.py, saved as transcripts_ytasr/<id>.json.
Word-timed lines (those containing <c> tags) are taken as segments.
Usage: .venv/bin/python -I scripts/collect/youtube/vtt_to_transcript.py
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/old/raw/youtube"
TS = re.compile(r"(\d+):(\d+):(\d+\.\d+)")


def sec(m):
    h, mi, s = m
    return int(h) * 3600 + int(mi) * 60 + float(s)


def parse(p):
    segs, start, end = [], None, None
    for line in p.read_text(errors="replace").splitlines():
        if "-->" in line:
            a, b = TS.findall(line)[:2]
            start, end = sec(a), sec(b)
        elif "<c>" in line and start is not None:
            text = re.sub(r"<[^>]+>", "", line).strip()
            if text and text != "[μουσική]":
                segs.append({"start": round(start, 2), "end": round(end, 2), "text": text,
                             "avg_logprob": -0.4, "no_speech_prob": 0.0, "words": []})
    return segs


def main():
    d = OUT / "transcripts_ytasr"
    d.mkdir(exist_ok=True)
    n = 0
    for p in sorted((OUT / "meta").glob("*.el-orig.vtt")):
        vid = p.name.split(".")[0]
        segs = parse(p)
        if segs:
            (d / f"{vid}.json").write_text(json.dumps(
                {"id": vid, "url": f"https://www.youtube.com/watch?v={vid}", "model": "youtube-auto-captions",
                 "segments": segs}, ensure_ascii=False, indent=0))
            n += 1
    print(n, "auto-caption transcripts")


if __name__ == "__main__":
    main()
