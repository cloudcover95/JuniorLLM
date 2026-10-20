"""Closed cycle with web3node absmean and agent update. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.closed_cycle import cycle
from ports.web3_absmean import feed
from ports.agent_mgr import update

def run(note: str = "cycle web3") -> dict:
    energy = feed(note)
    result = cycle(note)
    env = {"layer": "T0" if result["decision"]["keep"] else "T1", "max_tokens": 512}
    update("goldend", env, energy["value"], result["decision"].get("handoff", ""))
    body = {"energy": energy, "cycle": result, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "cycle_web3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
