"""Ternary efficiency check for the agent cycle. Local only."""
from __future__ import annotations
import json
import time
from pathlib import Path
from ports.trit_score import pack_trit

def check(n: int = 2000, note: str = "status note") -> dict:
    t0 = time.perf_counter()
    for _ in range(n):
        list(note[:64])
    base = (time.perf_counter() - t0) * 1000 / n
    t1 = time.perf_counter()
    for _ in range(n):
        pack_trit(note)
    trit = (time.perf_counter() - t1) * 1000 / n
    body = {
        "baseline_ms": round(base, 4),
        "trit_ms": round(trit, 4),
        "cheaper": trit < base,
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "ternary_efficiency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
