"""Trit receipt for JuniorSOL. Not an on-chain program."""
from __future__ import annotations

from ports.gaia_proto import handshake
from ports.flagstaff_balance import check


def receipt(note: str = "member share") -> dict:
    gate = check(note)
    hs = handshake(note, job="dash-viewport") if gate.get("ok") else {}
    return {
        "ok": gate.get("ok"),
        "votes": gate.get("votes"),
        "protocol": hs.get("protocol") or "goldend-osai-omega/1",
        "gamma": (hs.get("note") or {}).get("gamma"),
        "i2s_hex": (hs.get("note") or {}).get("i2s_hex"),
        "rpc": False,
        "program_id": None,
        "submit": False,
        "ledger": "local-jsonl",
    }
