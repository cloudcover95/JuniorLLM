"""Goldend closed cycle entry. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.closed_cycle import cycle

def run(note: str = "goldend cycle") -> dict:
    result = cycle(note)
    body = {"cycle": result, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "goldend_cycle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
