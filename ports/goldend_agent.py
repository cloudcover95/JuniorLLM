"""Goldend agent entry that uses compute-smart path. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.compute_smart import choose
from ports.hardware_probe import probe
from ports.junioros_agent import run as agent_run

def goldend(note: str = "goldend tick") -> dict:
    hw = probe()["hardware"]
    path = choose(note, hardware=hw)
    agent = agent_run(note, next_agent="goldend")
    body = {"path": path, "agent_envelope": agent.get("envelope"), "local": True}
    out = Path.home() / ".juniorhome" / "os" / "goldend_agent.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(goldend(), indent=2))
