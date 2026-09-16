"""Branching goldend tree. Kinds + parent from Hamming near-hits."""
from __future__ import annotations

import json
import math
from pathlib import Path

from ports.flagstaff_balance import check
from ports.trit_cache import hit, put

TREE = Path.home() / ".juniorhome" / "gaia_mesh" / "tree_dense.jsonl"
HTML = Path.home() / ".juniorhome" / "gaia_mesh" / "tree_dense.html"

KIND = {
    "issue": ("bind", "fail", "bug"),
    "artifact": ("obj", "png", "scan", "dxf", "mesh"),
    "experiment": ("bench", "tick", "gaia", "spine"),
    "belief": ("vault", "member", "share", "covenant"),
    "code": ("script", "port", "commit"),
}
STATE = {"n": 0, "scale": 1.0, "ids": {}}


def kind_of(note: str) -> str:
    n = (note or "").lower()
    for k, words in KIND.items():
        if any(w in n for w in words):
            return k
    return "artifact"


def _parent(near: list) -> str | None:
    if not near:
        return None
    return (near[0].get("note") or None)


def add(note: str) -> dict:
    gate = check(note)
    if not gate.get("ok"):
        return {"ok": False, "kind": "issue", "note": note, "failed": True}
    put(note)
    near = hit(note).get("near") or []
    STATE["n"] += 1
    STATE["scale"] = max(0.35, STATE["scale"] * 0.97)
    nid = f"n{STATE['n']}"
    parent = _parent(near)
    ang = (STATE["n"] * 2.399) % (2 * math.pi)
    r = 8 + STATE["n"] * 0.6
    node = {
        "id": nid,
        "note": note,
        "kind": kind_of(note),
        "parent": parent,
        "d": (near[0].get("d") if near else 0),
        "x": round(math.cos(ang) * r, 3),
        "y": round(math.sin(ang) * r, 3),
        "scale": STATE["scale"],
        "ok": True,
        "failed": False,
    }
    STATE["ids"][note] = nid
    TREE.parent.mkdir(parents=True, exist_ok=True)
    TREE.open("a", encoding="utf-8").write(json.dumps(node) + "\n")
    return node


def grow(notes: list[str] | None = None) -> dict:
    notes = notes or [
        "gaia spine",
        "journal field note",
        "home vault memory",
        "member share vault",
        "dxf title block",
        "scan jpeg title block",
        "buy oats",
        "buy oat",
        "home dash",
    ]
    rows = [add(n) for n in notes]
    _html(rows)
    return {"n": sum(1 for r in rows if r.get("ok")), "failed": sum(1 for r in rows if r.get("failed")), "scale": STATE["scale"], "path": str(TREE), "html": str(HTML)}


def _html(rows: list) -> None:
    pts = [r for r in rows if r.get("ok")]
    dots = ",".join(
        f"<circle cx='{250+r['x']*3:.1f}' cy='{200+r['y']*3:.1f}' r='6' fill='{_color(r['kind'])}'><title>{r['kind']}: {r['note']}</title></circle>"
        for r in pts
    )
    HTML.write_text(
        "<!doctype html><meta charset=utf-8><title>goldend tree</title>"
        f"<svg viewBox='0 0 500 400' style='background:#111'>{dots}</svg>",
        encoding="utf-8",
    )


def _color(k: str) -> str:
    return {
        "issue": "#4aa3ff",
        "artifact": "#e07a3d",
        "experiment": "#3dce9a",
        "belief": "#9b6bdb",
        "code": "#e0b84a",
    }.get(k, "#888")
