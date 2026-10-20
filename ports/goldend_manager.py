"""Goldend that reads the home manager result. Local only."""
from __future__ import annotations
import json
from pathlib import Path

HOME = Path.home() / ".juniorhome" / "os" / "home_manager.json"

def run() -> dict:
    body = {"keep": False, "synced": False, "local": True}
    if HOME.exists():
        try:
            data = json.loads(HOME.read_text(encoding="utf-8"))
            body = {"keep": data.get("keep", False), "synced": data.get("synced", False), "local": True}
        except Exception:
            pass
    out = Path.home() / ".juniorhome" / "os" / "goldend_manager.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
