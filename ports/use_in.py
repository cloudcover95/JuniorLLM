"""User jobs that are not CAD: media, notes, scan-to-vault."""
from __future__ import annotations

from pathlib import Path

from junior_bitnet.tp import tp_bitlinear
from ports.flagstaff_balance import check, guess
from ports.inject import write_vault

SAMPLES = {
    "media": "mp3 trim edit wav mix on the van node",
    "notes": "journal analysis expand this field note",
    "scan": "photocopy scan tiff store in home vault",
}


def ingest(vault: Path, kind: str = "notes") -> dict:
    note = SAMPLES[kind]
    bal = check(note)
    xs = [float(ord(c) % 97) for c in note[:16]]
    tp = tp_bitlinear(xs, xs[::-1] or [1.0], 2)
    row = write_vault(note, vault) if bal["ok"] else {"ok": False}
    return {
        "kind": kind,
        "guess": guess(note),
        "balance": bal["ok"],
        "tp_match": tp["match"],
        "area": row.get("area"),
        "inbox": row.get("inbox"),
        "ok": bool(row.get("ok")),
    }
