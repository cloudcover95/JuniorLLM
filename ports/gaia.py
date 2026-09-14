"""Gaia — local companion spine. Not Halo Cortana. Not a weight pull."""
from __future__ import annotations

import json
from pathlib import Path

from junior_bitnet.winsor import pack
from ports.flagstaff_balance import check
from ports.registry import LLMPort

GAIA = LLMPort(
    "JuniorGaia",
    "companion-spine",
    "ternary-1.58",
    "home-portrait",
    0.0,
    "Named HUD. Goldend spines. OBJ terrain + I2_S. No UE5 launch.",
)

DEFAULT = {"name": "Gaia", "pronouns": "they", "voice": "local", "mesh": "obj"}


def load_who(path: Path | None = None) -> dict:
    p = path or Path.home() / ".juniorhome" / "gaia.json"
    if p.is_file():
        raw = json.loads(p.read_text(encoding="utf-8"))
        out = dict(DEFAULT)
        out.update({k: raw[k] for k in DEFAULT if k in raw})
        return out
    return dict(DEFAULT)


def spine(note: str, who: dict | None = None) -> dict:
    who = who or load_who()
    q = pack([ord(c) % 13 - 6 for c in (note or "gaia")[:32]] or [0.2, -1.0, 0.4])
    gate = check(note or "gaia home", area="auto")
    bolts = [{"i": i, "trit": int(t), "lit": t != 0} for i, t in enumerate(q.get("trit") or [])]
    return {
        "port": GAIA.name,
        "who": who,
        "ok": gate.get("ok") and q.get("ok"),
        "gate": gate,
        "gamma": q.get("gamma"),
        "bolts": bolts,
        "omega": "obj",
        "ue5_launch": False,
        "download": False,
        "likeness": "original-goldend",
    }
