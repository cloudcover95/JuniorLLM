"""Agent management layer for Home brain / OSai. Local only.

Tracks agents, their last envelope, trit score, and handoff.
No model pull. No external trainer.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

AGENTS = Path.home() / ".juniorhome" / "os" / "agents.jsonl"
STATUS = Path.home() / ".juniorhome" / "os" / "agent_status.json"

def register(name: str, role: str, layer: str = "T0") -> dict[str, Any]:
    body = {
        "name": name,
        "role": role,
        "layer": layer,
        "ts": datetime.utcnow().isoformat(),
        "local": True,
    }
    AGENTS.parent.mkdir(parents=True, exist_ok=True)
    with AGENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(body) + "\n")
    return body

def update(name: str, envelope: dict, trit_score: float, handoff: str = "") -> dict[str, Any]:
    body = {
        "name": name,
        "envelope": envelope,
        "trit_score": trit_score,
        "handoff": handoff,
        "ts": datetime.utcnow().isoformat(),
        "local": True,
    }
    status = {}
    if STATUS.exists():
        try:
            status = json.loads(STATUS.read_text(encoding="utf-8"))
        except Exception:
            status = {}
    status[name] = body
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return body

def list_agents() -> list[dict]:
    if not AGENTS.exists():
        return []
    out = []
    for line in AGENTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def status() -> dict:
    if not STATUS.exists():
        return {"agents": {}, "local": True}
    try:
        return json.loads(STATUS.read_text(encoding="utf-8"))
    except Exception:
        return {"agents": {}, "local": True}
