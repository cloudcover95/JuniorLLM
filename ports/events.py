"""Pure event emit for headless checks. Local only.

Emits timestamped events. No UI. Assertable from a script.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

EVENTS = Path.home() / ".juniorhome" / "os" / "events.jsonl"

def emit(name: str, payload: dict[str, Any]) -> dict:
    body = {
        "name": name,
        "ts": datetime.utcnow().isoformat(),
        "payload": payload,
        "local": True,
    }
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(body) + "\n")
    return body

def assert_event(name: str, key: str, expected: Any) -> bool:
    """Headless check. True if the latest matching event has the expected value."""
    if not EVENTS.exists():
        return False
    lines = EVENTS.read_text(encoding="utf-8").strip().splitlines()
    for line in reversed(lines):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("name") == name and ev.get("payload", {}).get(key) == expected:
            return True
    return False
