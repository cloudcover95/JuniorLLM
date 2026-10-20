"""Production status for the agent cycle. Local only."""
from __future__ import annotations
import json
from pathlib import Path

def latest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def status() -> dict:
    base = Path.home() / ".juniorhome" / "os"
    body = {
        "goldend_retry": latest(base / "goldend_retry.json"),
        "web3": latest(base / "cycle_web3.json"),
        "handoff": latest(base / "osai_handoff.json"),
        "ternary": latest(base / "ternary_absmean.json"),
        "local": True,
    }
    out = base / "production_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(status(), indent=2))
