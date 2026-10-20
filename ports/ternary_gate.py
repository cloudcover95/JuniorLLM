"""Ternary gate for the production rail. Local only.

Keeps only if trit pack is cheaper and the retry ok.
"""
from __future__ import annotations
import json
from pathlib import Path
from ports.ternary_absmean import check as tern_check
from ports.agent_mgr import update

def gate(note: str = "gate") -> dict:
    tern = tern_check(note)
    keep = tern.get("cheaper", False)
    env = {"layer": "T0" if keep else "T1", "max_tokens": 512}
    update("production", env, tern.get("absmean", 0.0), "" if keep else "repack")
    body = {"keep": keep, "ternary": tern, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "ternary_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(gate(), indent=2))
