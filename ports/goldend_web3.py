"""Goldend web3 check entry. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.cycle_web3_check import check
from ports.ternary_absmean import check as tern_check

def run(note: str = "goldend web3") -> dict:
    result = check(note)
    tern = tern_check(note)
    body = {"check": result, "ternary": tern, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "goldend_web3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
