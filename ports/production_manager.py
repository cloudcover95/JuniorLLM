"""Production manager. Runs the full loop and writes one status. Local only."""
from __future__ import annotations
import json
from pathlib import Path

def run(note: str = "manager") -> dict:
    from ports.production_with_sync import run as cycle_sync
    from ports.goldend_sync import run as goldend
    result = cycle_sync(note)
    gold = goldend()
    body = {
        "keep": result["cycle"]["keep"],
        "synced": result["sync"]["updated"],
        "goldend_keep": gold["keep"],
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "production_manager.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
