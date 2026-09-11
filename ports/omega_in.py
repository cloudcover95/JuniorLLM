"""Omega-shaped sheet → Flagstaff balance + Home vault + llama sit-beside."""
from __future__ import annotations

from pathlib import Path

from junior_bitnet.tp import tp_bitlinear
from ports.enduser_llm import spec
from ports.flagstaff_balance import check, guess
from ports.inject import write_vault
from rails.linux.llama import plan as llama_plan

SAMPLE = {
    "title": "BRACKET",
    "rev": "A",
    "height": 8,
    "note": "dxf omega drawing missing elevation fix later",
}


def ingest(vault: Path, sheet: dict | None = None) -> dict:
    sheet = sheet or SAMPLE
    note = str(sheet.get("note") or sheet.get("title") or "omega")
    bal = check(note)
    card = spec()
    lp = llama_plan()
    xs = [float(ord(c) % 97) for c in note[:16]]
    tp = tp_bitlinear(xs, xs[::-1] or [1.0], 2)
    row = write_vault(note, vault) if bal["ok"] else {"ok": False, "votes": bal["votes"]}
    return {
        "guess": guess(note),
        "port": bal.get("port"),
        "balance": bal["ok"],
        "sheet": sheet,
        "llm": {"name": card["name"], "runtime": card["runtime"], "llama_ready": lp["ready"]},
        "tp_match": tp["match"],
        "inject": {"ok": row.get("ok"), "area": row.get("area"), "inbox": row.get("inbox")},
    }
