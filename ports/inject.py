"""Digest a note into the inbox that matches its Home domain."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ports.flagstaff import assemble
from ports.flagstaff_balance import check

INBOX = {
    "climbs": "stonefield_inbox.jsonl",
    "stonefield": "stonefield_inbox.jsonl",
    "flagstaff": "stonefield_inbox.jsonl",
    "xanadu": "stonefield_inbox.jsonl",
    "stock": "stock_nodes.jsonl",
    "cad": "cad_inbox.jsonl",
    "omega": "cad_inbox.jsonl",
    "os": "os_inbox.jsonl",
    "van": "van_inbox.jsonl",
    "llm": "llm_inbox.jsonl",
    "home": "home_inbox.jsonl",
    "vault": "home_inbox.jsonl",
}


def _id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def digest(note: str, *, area: str = "auto", consent: bool = True, private: bool = False) -> dict:
    bal = check(note, area=area, consent=consent, private=private)
    if not bal["ok"]:
        return {"ok": False, "reason": "balance", "votes": bal["votes"], "area": bal.get("area")}
    tf = bal["tf"]
    area = bal["area"]
    nid = _id(tf["text"])
    node = {
        "id": nid,
        "area": area,
        "text": tf["text"],
        "port": tf["port"],
        "fusion_y": tf["fusion_y"],
        "consent": consent,
        "private": private,
        "kind": "custom",
    }
    return {"ok": True, "votes": bal["votes"], "area": area, "inbox": INBOX.get(area, "home_inbox.jsonl"), "node": node, "ctx_total": assemble(note)["total"]}


def write_vault(note: str, vault: Path, **kw) -> dict:
    row = digest(note, **kw)
    vault = Path(vault)
    vault.mkdir(parents=True, exist_ok=True)
    if not row.get("ok"):
        (vault / "inject_denied.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        return row
    dest = vault / row["inbox"]
    with dest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row["node"]) + "\n")
    with (vault / "nodes.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row["node"]) + "\n")
    return row
