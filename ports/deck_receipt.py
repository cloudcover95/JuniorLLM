"""Append a receipt only when deck_trit Flagstaff passes."""
from __future__ import annotations

import json
from pathlib import Path

from ports.deck_trit import load

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "deck_receipt.jsonl"


def receipt() -> dict:
    row = load()
    if not row.get("flagstaff"):
        return {"appended": False, "row": row}
    MESH.parent.mkdir(parents=True, exist_ok=True)
    line = {"sha3": row.get("sha3"), "audio_n": row.get("audio_n"), "ok": True}
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    return {"appended": True, "row": row}


if __name__ == "__main__":
    print(json.dumps(receipt(), indent=2))
