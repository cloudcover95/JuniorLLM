"""Efficiency bench for trit pack and RLVR envelope. Local only. No model pull."""
from __future__ import annotations
import json
import time
from pathlib import Path
from ports.trit_score import pack_trit, trit_reward
from ports.rlvr import scale_envelope, Reward

def bench_trit(n: int = 1000, note: str = "status note for envelope") -> dict:
    t0 = time.perf_counter()
    for _ in range(n):
        pack_trit(note)
    ms = (time.perf_counter() - t0) * 1000 / n
    score, passed, detail = trit_reward(note)
    return {"op": "trit_pack", "n": n, "ms_per": round(ms, 4), "score": score, "passed": passed, "detail": detail}

def bench_envelope(n: int = 1000) -> dict:
    rewards = [Reward("ok", 0.9, True)]
    t0 = time.perf_counter()
    for _ in range(n):
        scale_envelope(rewards, note_len=64, hardware="cpu")
    ms = (time.perf_counter() - t0) * 1000 / n
    env = scale_envelope(rewards, note_len=64, hardware="cpu")
    return {"op": "envelope", "n": n, "ms_per": round(ms, 4), "layer": env.layer, "max_tokens": env.max_tokens}

def run() -> dict:
    body = {"trit": bench_trit(), "envelope": bench_envelope(), "local": True}
    out = Path.home() / ".juniorhome" / "os" / "bench_rlvr.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
