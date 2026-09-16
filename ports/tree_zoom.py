"""Inference tree of cache/imager hits. Zoom-out is a scale, not a video."""
from __future__ import annotations

import json
from pathlib import Path

from ports.imager import image
from ports.receipt_cache import issue
from ports.trit_cache import hit, put

TREE = Path.home() / ".juniorhome" / "gaia_mesh" / "tree.jsonl"
STATE = {"scale": 1.0, "nodes": 0}


def _zoom() -> float:
    STATE["scale"] = max(0.35, STATE["scale"] * 0.97)
    STATE["nodes"] += 1
    return STATE["scale"]


def tick(note: str, *, neighbor: str | None = None) -> dict:
    put(note)
    img = image(note)
    rec = issue(note, neighbor) if img.get("ok") else {"ok": False}
    near = hit(neighbor or note)
    node = {
        "note": note,
        "ok": bool(img.get("ok") and rec.get("ok")),
        "scale": _zoom() if img.get("ok") else STATE["scale"],
        "nodes": STATE["nodes"],
        "exact": near.get("exact"),
        "near": near.get("near"),
        "gamma": img.get("gamma"),
        "orient": "portrait" if STATE["scale"] < 0.7 else "landscape",
        "ue5": False,
        "swarm": False,
        "download": False,
    }
    TREE.parent.mkdir(parents=True, exist_ok=True)
    TREE.open("a", encoding="utf-8").write(json.dumps(node) + "\n")
    return node


def grow(notes: list[str] | None = None) -> dict:
    notes = notes or [
        "journal field note",
        "home vault memory",
        "buy oats",
        "dxf title block",
        "gaia spine",
    ]
    rows = [tick(n) for n in notes]
    return {"n": len(rows), "scale": STATE["scale"], "path": str(TREE), "rows": rows}
