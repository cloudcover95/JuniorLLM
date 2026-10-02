"""Read the Home process table. boot stays false."""
from __future__ import annotations

import json
from pathlib import Path

TABLE = Path.home() / ".juniorhome" / "os" / "ps.jsonl"


def read() -> dict:
    if not TABLE.exists():
        return {"ok": False, "reason": "no ps", "boot": False}
    lines = TABLE.read_text(encoding="utf-8").strip().splitlines()
    last = json.loads(lines[-1]) if lines else {}
    return {"ok": True, "n": len(lines), "last": last, "boot": False, "bind": "127.0.0.1"}
