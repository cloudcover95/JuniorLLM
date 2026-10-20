"""OSai handoff from Home full stack result. Local only."""
from __future__ import annotations
import json
from pathlib import Path

HOME = Path.home() / ".juniorhome" / "os" / "home_full_stack.json"

def handoff() -> dict:
    body = {"keep": False, "handoff": "repack", "local": True}
    if HOME.exists():
        try:
            data = json.loads(HOME.read_text(encoding="utf-8"))
            if data.get("asserted") and data.get("energy", 0) >= 0.3:
                body = {"keep": True, "handoff": "", "energy": data.get("energy"), "local": True}
            else:
                body = {"keep": False, "handoff": "repack", "energy": data.get("energy"), "local": True}
        except Exception:
            pass
    out = Path.home() / ".juniorhome" / "os" / "osai_handoff.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(handoff(), indent=2))
