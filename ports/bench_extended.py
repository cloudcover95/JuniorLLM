"""Extended efficiency bench: trit pack vs baseline, envelope cost, goldend cycle. Local only."""
from __future__ import annotations
import json
import time
from pathlib import Path
from ports.trit_score import pack_trit, trit_reward
from ports.compute_smart import choose

def baseline_pack(text: str) -> list[int]:
    return [ord(c) for c in text[:64]]

def bench_compare(n: int = 2000, note: str = "status note for envelope") -> dict:
    t0 = time.perf_counter()
    for _ in range(n):
        baseline_pack(note)
    base_ms = (time.perf_counter() - t0) * 1000 / n
    t1 = time.perf_counter()
    for _ in range(n):
        pack_trit(note)
    trit_ms = (time.perf_counter() - t1) * 1000 / n
    score, passed, detail = trit_reward(note)
    return {
        "baseline_ms": round(base_ms, 4),
        "trit_ms": round(trit_ms, 4),
        "ratio": round(base_ms / trit_ms, 2) if trit_ms else 0,
        "score": score,
        "passed": passed,
        "detail": detail,
    }

def bench_choose(n: int = 1000) -> dict:
    t0 = time.perf_counter()
    for _ in range(n):
        choose("os tick", hardware="cpu")
    ms = (time.perf_counter() - t0) * 1000 / n
    path = choose("os tick", hardware="cpu")
    return {"ms_per": round(ms, 4), "layer": path["layer"], "max_tokens": path["max_tokens"]}

def run() -> dict:
    body = {"compare": bench_compare(), "choose": bench_choose(), "local": True}
    out = Path.home() / ".juniorhome" / "os" / "bench_extended.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
