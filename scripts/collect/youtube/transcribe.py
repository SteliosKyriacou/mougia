"""Download audio-only, isolate vocals, transcribe with faster-whisper large-v3.

Pipeline per video id:
  yt-dlp worst/low-bitrate audio -> audio_tmp/<id>.<ext>
  ffmpeg -> 44.1k wav -> demucs htdemucs --two-stems vocals (CPU)
  vocals -> 16k mono; energy-based "sung region" detection on the vocal stem
  (whisper's Silero VAD drops singing over violin; whisper without VAD
  hallucinates "Υπότιτλοι AUTHORWAVE" and skips 30 s windows)
  regions packed into <=28 s clips -> faster-whisper (language=el,
  word timestamps, clip_timestamps, hallucination tokens suppressed)
  -> transcripts/<id>.json ; all audio deleted.

GPU: this machine's GPU was occupied by another process (vLLM, ~11 GB), so the
default is CPU int8. Pass --device cuda when the GPU is free.

Usage: .venv/bin/python -I scripts/collect/youtube/transcribe.py --ids-file F
       [--model large-v3] [--device cpu] [--beam 1] [--max-hours 6]
"""
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/raw/youtube"
AUDIO = OUT / "audio_tmp"
TR = OUT / "transcripts"
BIN = ROOT / ".venv/bin"
YTDLP = str(BIN / "yt-dlp")
FFMPEG = str(BIN / "ffmpeg")
ENV = dict(os.environ, PATH=f"{BIN}:{os.environ.get('PATH', '')}")

HALLU_WORDS = [" Υπότιτλοι", "Υπότιτλοι", " AUTHORWAVE", "AUTHORWAVE", " Μουσική", "Μουσική"]


def download(vid):
    AUDIO.mkdir(parents=True, exist_ok=True)
    cmd = [YTDLP, "--js-runtimes", "node", "-f", "worstaudio[abr>=45]/bestaudio[abr<=96]/bestaudio",
           "--sleep-requests", "1", "--no-progress", "--no-playlist", "--max-filesize", "300M",
           "-o", str(AUDIO / "%(id)s.%(ext)s"), "--", vid]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=ENV)
    if r.returncode != 0:
        if re.search(r"confirm you.re not a bot|HTTP Error 429", r.stderr):
            sys.exit("YouTube bot-check / rate limit; stopping.")
        print("ERR dl", vid, r.stderr[-300:], file=sys.stderr)
        return None
    fs = [f for f in AUDIO.glob(f"{vid}.*") if not f.name.endswith((".part", ".wav"))]
    return fs[0] if fs else None


def separate(src, vid, jobs):
    wav = AUDIO / f"{vid}.44k.wav"
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-i", str(src), "-ar", "44100", "-ac", "2", str(wav)],
                   check=True)
    sep = AUDIO / f"sep_{vid}"
    subprocess.run([sys.executable, "-I", "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs",
                    "-d", "cpu", "-j", str(jobs), "-o", str(sep), str(wav)],
                   check=True, capture_output=True, env=ENV)
    voc = next(sep.rglob("vocals.wav"))
    out = AUDIO / f"{vid}.voc16k.wav"
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-i", str(voc), "-ar", "16000", "-ac", "1", str(out)],
                   check=True)
    shutil.rmtree(sep)
    wav.unlink()
    return out


def sung_regions(a, sr=16000):
    hop = sr // 10
    n = len(a) // hop
    e = np.sqrt(np.mean(a[: n * hop].reshape(n, hop) ** 2, axis=1))
    db = 20 * np.log10(e + 1e-9)
    lo, hi = np.percentile(db, 30), np.percentile(db, 95)
    on = db > lo + 0.5 * (hi - lo)
    regs, i = [], 0
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            regs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for r in regs:
        if merged and r[0] - merged[-1][1] < 15:  # close gaps < 1.5 s
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(s / 10, e / 10) for s, e in merged if e - s >= 10]  # drop < 1 s


def pack(regions, maxlen=28.0, maxgap=3.0):
    clips = []
    for s, e in regions:
        s, e = max(0.0, s - 0.3), e + 0.3
        while e - s > maxlen:
            clips.append([s, s + maxlen]); s += maxlen
        if clips and s - clips[-1][1] < maxgap and e - clips[-1][0] <= maxlen:
            clips[-1][1] = e
        else:
            clips.append([s, e])
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--no-demucs", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    ids = [l.split()[0] for l in open(a.ids_file) if l.strip() and not l.startswith("#")]
    if a.device == "cuda":
        model = WhisperModel(a.model, device="cuda", compute_type="int8_float16")
    else:
        model = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=a.threads)
    tok = model.hf_tokenizer
    sup = set()
    for w in HALLU_WORDS:
        ids_ = tok.encode(w, add_special_tokens=False).ids
        sup.update(ids_ if "AUTH" in w else ids_[:1])
    TR.mkdir(parents=True, exist_ok=True)
    total = sum(json.loads(p.read_text()).get("duration", 0) for p in TR.glob("*.json"))
    for vid in ids:
        out = TR / f"{vid}.json"
        if out.exists():
            continue
        if total > a.max_hours * 3600:
            print("audio budget reached", file=sys.stderr)
            break
        src = download(vid)
        if not src:
            continue
        t0 = time.time()
        try:
            f = src if a.no_demucs else separate(src, vid, 4)
            audio = decode_audio(str(f), sampling_rate=16000)
            regions = sung_regions(audio)
            clips = pack(regions)
            t1 = time.time()
            segs, info = model.transcribe(audio, language="el", word_timestamps=True, vad_filter=False,
                                          clip_timestamps=[x for c in clips for x in c] or "0",
                                          suppress_tokens=[-1] + sorted(sup),
                                          condition_on_previous_text=False, beam_size=a.beam,
                                          # sung repetitions trip the compression-ratio fallback and
                                          # cause 5x re-decoding / loops: cap fallback and length
                                          temperature=[0.0, 0.4], compression_ratio_threshold=3.0,
                                          max_new_tokens=120)
            S = []
            for s in segs:
                S.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                          "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3),
                          "compression_ratio": round(s.compression_ratio, 2),
                          "words": [[round(w.start, 2), round(w.end, 2), w.word, round(w.probability, 3)]
                                    for w in (s.words or [])]})
        finally:
            for p in AUDIO.glob(f"{vid}*"):
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
        dur = len(audio) / 16000
        rec = {"id": vid, "url": f"https://www.youtube.com/watch?v={vid}", "model": a.model,
               "beam_size": a.beam, "device": a.device, "vocal_separation": None if a.no_demucs else "demucs htdemucs",
               "duration": round(dur, 1), "sung_regions": regions, "clips": clips,
               "sep_s": round(t1 - t0, 1), "asr_s": round(time.time() - t1, 1), "segments": S}
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=0))
        total += dur
        print(f"{vid} dur={dur:.0f}s sep={rec['sep_s']}s asr={rec['asr_s']}s segs={len(S)} total_h={total/3600:.2f}",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
