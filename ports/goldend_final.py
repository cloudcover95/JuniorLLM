"""Goldend that reads the final agent. Local only."""
from __future__ import annotations
import json
from pathlib import Path

AGENT = Path.home() / ".juniorhome" / "os" / "final_agent.json"

def run() -> dict:
    body = {"updated": False, "ok": False, "local": True}
    if AGENT.exists():
        try:
            data = json.loads(AGENT.read_text(encoding="utf-8"))
            body = {"updated": data.get("updated", False), "ok": data.get("ok", False), "local": True}
        except Exception:
            pass
    out = Path.home() / ".juniorhome" / "os" / "goldend_final.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
