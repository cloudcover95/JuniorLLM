"""Goldend path that uses the efficient trit math. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.trit_math import once
from ports.osai_terraform import decide

def path(note: str = "goldend path") -> dict:
    math = once(note)
    decision = decide(note)
    affirm = math["passed"] and decision.get("keep", False)
    body = {"math": math, "decision": decision, "affirm": affirm, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "goldend_path.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(path(), indent=2))
