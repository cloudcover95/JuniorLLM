"""Production cycle with ternary gate. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.ternary_gate import gate
from ports.goldend_retry import run as retry_run

def cycle(note: str = "production cycle") -> dict:
    g = gate(note)
    r = retry_run(note)
    keep = g["keep"] and r["retry"]["ok"]
    body = {"gate": g, "retry": r, "keep": keep, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "production_cycle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(cycle(), indent=2))
