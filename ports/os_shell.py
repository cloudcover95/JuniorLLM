"""Read the Home session. boot stays false."""
from __future__ import annotations

import json
from pathlib import Path

SESSION = Path.home() / ".juniorhome" / "os" / "session.json"
TABLE = Path.home() / ".juniorhome" / "os" / "ps.jsonl"


def read() -> dict:
    if not SESSION.exists():
        return {"ok": False, "reason": "no session", "boot": False}
    session = json.loads(SESSION.read_text(encoding="utf-8"))
    n = 0
    if TABLE.exists():
        n = len(TABLE.read_text(encoding="utf-8").strip().splitlines())
    return {"ok": True, "session": session, "ps": n, "boot": False, "bind": "127.0.0.1"}
