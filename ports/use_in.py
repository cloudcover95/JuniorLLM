"""Any use-case note through Flagstaff + TP + llama sit-beside."""
from __future__ import annotations

from pathlib import Path

from junior_bitnet.tp import tp_bitlinear
from ports.enduser_llm import spec
from ports.flagstaff_balance import check, guess
from ports.inject import write_vault
from rails.linux.llama import plan as llama_plan


def ingest(vault: Path, note: str) -> dict:
    bal = check(note)
    card = spec()
    xs = [float(ord(c) % 97) for c in note[:16]]
    tp = tp_bitlinear(xs, xs[::-1] or [1.0], 2)
    row = write_vault(note, vault) if bal["ok"] else {"ok": False}
    return {
        "guess": guess(note),
        "port": bal.get("port"),
        "balance": bal["ok"],
        "llm": {"name": card["name"], "runtime": card["runtime"], "llama_ready": llama_plan()["ready"]},
        "tp_match": tp["match"],
        "inject": {"ok": row.get("ok"), "area": row.get("area"), "inbox": row.get("inbox")},
    }
