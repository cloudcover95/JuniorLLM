"""Sandbox bench for the inference engine. Local only."""
from __future__ import annotations
import json
import time
from pathlib import Path
from ports.inference_engine import record, latest

def bench(n: int = 500) -> dict:
    notes = ["os tick", "goldend status note", "home rail check", "web3node sample"]
    t0 = time.perf_counter()
    for i in range(n):
        record(notes[i % len(notes)], source="bench")
    ms = (time.perf_counter() - t0) * 1000 / n
    body = {"n": n, "ms_per": round(ms, 4), "latest": latest(3), "local": True}
    out = Path.home() / ".juniorhome" / "os" / "inference_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(bench(), indent=2))
