"""OSai workflow that affirms the goldend path. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.goldend_path import path
from ports.agent_mgr import update

def affirm(note: str = "osai affirm") -> dict:
    result = path(note)
    layer = "T0" if result["affirm"] else "T1"
    update("goldend", {"layer": layer, "max_tokens": 512}, result["math"]["score"], "" if result["affirm"] else "repack")
    body = {"affirm": result["affirm"], "layer": layer, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "osai_affirm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(affirm(), indent=2))
