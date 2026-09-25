"""MemSys-shaped jsonl. Not a palace. Not Faiss."""
from __future__ import annotations

import json
from pathlib import Path

LAKE = Path.home() / ".juniorhome" / "mem" / "notes.jsonl"


def remember(note: str, kind: str = "gaia") -> dict:
    LAKE.parent.mkdir(parents=True, exist_ok=True)
    row = {"note": note, "kind": kind, "faiss": False, "palace": False}
    with LAKE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return {"wrote": True, "path": str(LAKE)}
