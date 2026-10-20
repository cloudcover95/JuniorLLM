"""Closed production loop. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.final_agent import sync
from ports.energy_check import check

def loop(note: str = "closed") -> dict:
    energy = check(note)
    agent = sync()
    body = {"energy_ok": energy["ok"], "agent_updated": agent["updated"], "local": True}
    out = Path.home() / ".juniorhome" / "os" / "closed_loop.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(loop(), indent=2))
