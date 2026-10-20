"""Bench the production with sync cycle. Local only."""
from __future__ import annotations
import json
import time
from pathlib import Path
from ports.production_with_sync import run

def bench(n: int = 5) -> dict:
    t0 = time.perf_counter()
    last = None
    for _ in range(n):
        last = run("bench")
    ms = (time.perf_counter() - t0) * 1000 / n
    body = {"n": n, "ms_per": round(ms, 4), "last_keep": last["cycle"]["keep"] if last else None, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "sync_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(bench(), indent=2))
