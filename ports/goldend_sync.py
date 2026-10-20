"""Goldend that reads the sync rail result. Local only."""
from __future__ import annotations
import json
from pathlib import Path

RAIL = Path.home() / ".juniorhome" / "os" / "home_sync_rail.json"

def run() -> dict:
    body = {"keep": False, "synced": False, "local": True}
    if RAIL.exists():
        try:
            data = json.loads(RAIL.read_text(encoding="utf-8"))
            body = {"keep": data.get("keep", False), "synced": data.get("synced", False), "local": True}
        except Exception:
            pass
    out = Path.home() / ".juniorhome" / "os" / "goldend_sync.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
