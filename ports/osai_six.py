"""Exercise the six runtimes. Same compute as one handshake each."""
from __future__ import annotations

from ports.osai_engines import run

NOTES = {
    "vault": "vault seal porch light",
    "media": "mp3 mix edit",
    "notes": "journal field note",
    "stock": "ticker node book",
    "cad": "dxf title block",
    "os": "junioros llama",
}


def six() -> dict:
    rows = {k: run(k, n) for k, n in NOTES.items()}
    return {"n": 6, "ok": all(r.get("ok") for r in rows.values()), "rows": rows, "new_so": False}
