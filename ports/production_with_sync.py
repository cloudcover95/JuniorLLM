"""Production cycle with agent sync. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.production_cycle import cycle
from ports.agent_sync import sync

def run(note: str = "production with sync") -> dict:
    result = cycle(note)
    synced = sync()
    body = {"cycle": result, "sync": synced, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "production_with_sync.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
