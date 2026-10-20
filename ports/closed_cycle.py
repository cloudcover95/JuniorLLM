"""Closed cycle: Home rail then OSai handoff. Local only."""
from __future__ import annotations
import json
import time
from pathlib import Path

def cycle(note: str = "cycle") -> dict:
    t0 = time.perf_counter()
    from scripts.home_full_stack import rail
    from ports.osai_handoff import handoff
    rail_result = rail(note)
    decision = handoff()
    ms = (time.perf_counter() - t0) * 1000
    body = {
        "rail": rail_result,
        "decision": decision,
        "ms": round(ms, 4),
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "closed_cycle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(cycle(), indent=2))
