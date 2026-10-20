"""Goldend efficiency report for Home rails. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.goldend_agent import goldend
from ports.bench_rlvr import run as bench_run

def report(note: str = "goldend efficiency") -> dict:
    bench = bench_run()
    agent = goldend(note)
    body = {
        "bench_trit_ms": bench["trit"]["ms_per"],
        "bench_envelope_ms": bench["envelope"]["ms_per"],
        "agent_layer": agent["path"]["layer"],
        "agent_score": agent["path"]["trit_score"],
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "goldend_efficiency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
