"""Self-check for the web3 cycle. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.cycle_web3 import run

def check(note: str = "self check") -> dict:
    result = run(note)
    ok = "energy" in result and "cycle" in result and "decision" in result["cycle"]
    body = {"ok": ok, "energy": result["energy"]["value"], "keep": result["cycle"]["decision"]["keep"], "local": True}
    out = Path.home() / ".juniorhome" / "os" / "cycle_web3_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
