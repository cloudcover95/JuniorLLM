"""OSai agent manager. Registers agents and updates from RLVR cycles. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.agent_mgr import register, update, status
from ports.compute_smart import choose
from ports.hardware_probe import probe

def ensure() -> None:
    register("junioros", "os tick", "T0")
    register("goldend", "efficiency", "T0")
    register("home", "status", "T0")

def tick(name: str, note: str) -> dict:
    hw = probe()["hardware"]
    path = choose(note, hardware=hw)
    return update(name, {"layer": path["layer"], "max_tokens": path["max_tokens"]}, path["trit_score"], path.get("handoff", ""))

def report() -> dict:
    ensure()
    tick("junioros", "os tick")
    tick("goldend", "goldend tick")
    body = status()
    out = Path.home() / ".juniorhome" / "os" / "osai_agents.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
