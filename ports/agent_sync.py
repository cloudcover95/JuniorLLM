"""Update agent status from the production summary. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.agent_mgr import update

SUMMARY = Path.home() / ".juniorhome" / "os" / "production_summary.json"

def sync() -> dict:
    body = {"updated": False, "local": True}
    if SUMMARY.exists():
        try:
            data = json.loads(SUMMARY.read_text(encoding="utf-8"))
            cycle = data.get("cycle") or {}
            keep = cycle.get("keep", False)
            env = {"layer": "T0" if keep else "T1", "max_tokens": 512}
            update("production", env, 1.0 if keep else 0.0, "" if keep else "repack")
            body = {"updated": True, "keep": keep, "local": True}
        except Exception:
            pass
    out = Path.home() / ".juniorhome" / "os" / "agent_sync.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(sync(), indent=2))
