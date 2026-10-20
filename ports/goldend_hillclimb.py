"""Goldend hill-climb entry. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.hillclimb import climb, trim_mutate
from ports.compute_smart import choose
from ports.hardware_probe import probe

def run(note: str = "goldend status note for envelope", held_out: str = "held out note") -> dict:
    hw = probe()["hardware"]
    path = choose(note, hardware=hw)
    result = climb(note, trim_mutate, held_out=held_out, rounds=3)
    body = {
        "path": path,
        "climb": {"best_score": result["best_score"], "rounds": len(result["history"])},
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "goldend_hillclimb.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
