"""Persistent run logging for Mougia training — everything the paper and the browser's Training tab need.

Layout of one run (all append-only JSONL, flushed on every write so the browser sees it live):

    runs/<run_id>/
      config.json     model / connectome / data / optimizer settings, seed, git hash, host, library versions
      status.json     {"state": "running|finished|failed|stopped", "updated_at", "step", "elapsed_s", "note"}
      metrics.jsonl   one line per logged training step: step, elapsed_s, phase, train_loss, lr, grad_norm, ...
      eval.jsonl      one line per evaluation: step, elapsed_s, phase, val_loss, val_ppl, meter_rate, rhyme_rate, cy_rate, ...
      samples.jsonl   one line per generated reply: step, elapsed_s, phase, prompt, reply, scores
      checkpoints/    model checkpoints (not read by the browser)

`elapsed_s` is wall-clock *training* time since the run started (excluding time spent generating samples),
so "every 15 minutes of training" means the same thing across runs.
"""
import json
import os
import platform
import socket
import subprocess
import time

SAMPLE_EVERY_S = 15 * 60


def _git_hash():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain"], capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return None


def _env():
    env = {"python": platform.python_version(), "host": socket.gethostname()}
    try:
        import torch
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return env


class RunLogger:
    def __init__(self, run_id, config, runs_dir=None, sample_every_s=SAMPLE_EVERY_S):
        self.runs_dir = runs_dir or os.environ.get("MOUGIA_RUNS") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
        self.dir = os.path.join(self.runs_dir, run_id)
        os.makedirs(os.path.join(self.dir, "checkpoints"), exist_ok=True)
        self.run_id = run_id
        self.sample_every_s = sample_every_s
        self._t0 = time.time()
        self._paused = 0.0
        self._last_sample = -1e18
        self._files = {}
        self.step = 0
        cfg = dict(config)
        cfg.setdefault("run_id", run_id)
        cfg["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        cfg["git"] = _git_hash()
        cfg["env"] = _env()
        cfg["sample_every_s"] = sample_every_s
        with open(os.path.join(self.dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        self.status("running")

    # --- time -----------------------------------------------------------------------------------
    def elapsed(self):
        """Training seconds since start, excluding time spent inside sampling()."""
        return time.time() - self._t0 - self._paused

    # --- writers --------------------------------------------------------------------------------
    def _write(self, name, rec):
        f = self._files.get(name)
        if f is None:
            f = self._files[name] = open(os.path.join(self.dir, name), "a", encoding="utf-8")
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()

    def _base(self, step, phase):
        self.step = step
        return {"step": step, "elapsed_s": round(self.elapsed(), 2), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "phase": phase}

    def metrics(self, step, phase, **values):
        """Training-step metrics, e.g. train_loss, lr, grad_norm, tokens_per_s, gpu_mem_gb, mean_rate."""
        self._write("metrics.jsonl", {**self._base(step, phase), **_clean(values)})

    def eval(self, step, phase, **values):
        """Held-out metrics, e.g. val_loss, val_ppl, meter_rate, rhyme_rate, cy_rate."""
        self._write("eval.jsonl", {**self._base(step, phase), **_clean(values)})
        self.status("running")

    def should_sample(self):
        return self.elapsed() - self._last_sample >= self.sample_every_s

    def sampling(self):
        """Context manager: time spent generating samples is excluded from elapsed training time."""
        logger = self

        class _Pause:
            def __enter__(self):
                self.t = time.time()

            def __exit__(self, *exc):
                logger._paused += time.time() - self.t
                logger._last_sample = logger.elapsed()

        return _Pause()

    def sample(self, step, phase, prompt, reply, **scores):
        """One generated reply. prompt/reply are couplets as 'line1\\nline2'."""
        self._write("samples.jsonl", {**self._base(step, phase), "prompt": prompt, "reply": reply, **_clean(scores)})

    def status(self, state, note=None):
        rec = {"state": state, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "step": self.step,
               "elapsed_s": round(self.elapsed(), 2), "note": note}
        tmp = os.path.join(self.dir, "status.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        os.replace(tmp, os.path.join(self.dir, "status.json"))

    def close(self, state="finished", note=None):
        self.status(state, note)
        for f in self._files.values():
            f.close()
        self._files = {}


def _clean(d):
    out = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float):
            v = round(v, 6) if v == v and abs(v) != float("inf") else None
        out[k] = v
    return out
