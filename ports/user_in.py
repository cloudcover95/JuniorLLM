"""Any user-side text. Hints optional."""
from __future__ import annotations

from pathlib import Path

from junior_bitnet.tp import tp_bitlinear
from ports.flagstaff_balance import check, guess
from ports.inject import write_vault


def ingest(vault: Path, note: str) -> dict:
    bal = check(note)
    xs = [float(ord(c) % 97) for c in (note or "x")[:16]]
    tp = tp_bitlinear(xs, xs[::-1] or [1.0], 2)
    row = write_vault(note, vault) if bal["ok"] else {"ok": False}
    return {
        "guess": guess(note),
        "balance": bal["ok"],
        "tp_match": tp["match"],
        "area": row.get("area"),
        "inbox": row.get("inbox"),
        "ok": bool(row.get("ok")),
        "port": bal.get("port"),
    }
