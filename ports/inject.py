"""Digest a user note → StoneField + Stock after Flagstaff balance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ports.flagstaff import assemble
from ports.flagstaff_balance import check


def _id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def digest(note: str, *, area: str = "flagstaff", consent: bool = True, private: bool = False) -> dict:
    bal = check(note, area=area, consent=consent, private=private)
    if not bal["ok"]:
        return {"ok": False, "reason": "balance", "votes": bal["votes"]}
    tf = bal["tf"]
    ctx = assemble(note)
    nid = _id(tf["text"])
    return {
        "ok": True,
        "votes": bal["votes"],
        "stone": {
            "id": nid,
            "area": area,
            "beta": tf["text"],
            "port": tf["port"],
            "fusion_y": tf["fusion_y"],
            "consent": consent,
            "private": private,
        },
        "stock": {
            "id": "node-" + nid,
            "kind": "custom",
            "label": tf["text"][:48],
            "src": "stonefield",
            "fusion_y": tf["fusion_y"],
        },
        "ctx_total": ctx["total"],
    }


def write_vault(note: str, vault: Path, **kw) -> dict:
    row = digest(note, **kw)
    vault = Path(vault)
    vault.mkdir(parents=True, exist_ok=True)
    if not row.get("ok"):
        (vault / "inject_denied.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        return row
    stones = vault / "stonefield_inbox.jsonl"
    stocks = vault / "stock_nodes.jsonl"
    with stones.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row["stone"]) + "\n")
    with stocks.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row["stock"]) + "\n")
    return row
