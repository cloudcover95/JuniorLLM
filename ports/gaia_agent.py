"""Deterministic Gaia agent: gate → terrain → UE5 spec. No LLM download."""
from __future__ import annotations

from pathlib import Path

from ports.flagstaff_balance import check
from ports.gaia import spine
from ports.gaia_terrain import write as terrain_write
from ports.ue5_port import write as ue5_write


def run(note: str, out_dir: Path, n: int = 32) -> dict:
    steps = []
    gate = check(note, area="auto")
    steps.append({"step": "flagstaff", "ok": gate.get("ok")})
    if not gate.get("ok"):
        return {"ok": False, "steps": steps, "ue5_launch": False}
    who = spine(note)
    steps.append({"step": "spine", "ok": who.get("ok"), "n_bolts": len(who.get("bolts") or [])})
    terr = terrain_write(out_dir, note, n)
    steps.append({"step": "terrain", "ok": terr.get("ok"), "faces": terr.get("faces")})
    ue = ue5_write(out_dir, note, n)
    steps.append({"step": "ue5_port", "ok": True, "launch": ue.get("launch")})
    return {"ok": all(s["ok"] for s in steps), "steps": steps, "out": str(out_dir), "ue5_launch": False, "download": False}
