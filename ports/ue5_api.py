"""UE5-shaped vectors/controls. Writes JSON. Does not import unreal. Does not launch."""
from __future__ import annotations

import json
from pathlib import Path

from ports.ue5_port import scene


def vec(x: float, y: float, z: float) -> dict:
    return {"x": float(x), "y": float(y), "z": float(z)}


def rot(p: float, y: float, r: float) -> dict:
    return {"pitch": float(p), "yaw": float(y), "roll": float(r)}


def controls(orient: str = "landscape") -> dict:
    return {
        "camera": {"loc": vec(0, -500, 300), "rot": rot(-20, 0, 0)},
        "light": {"loc": vec(0, 0, 1000), "rot": rot(-45, 45, 0), "intensity": 8.0},
        "terrain": {"loc": vec(0, 0, 0), "rot": rot(0, 0, 0)},
        "tape": {"loc": vec(120, 0, 40)},
        "drive": {"loc": vec(80, 80, 0)},
        "orient": orient,
        "launch": False,
    }


def board(note: str = "gaia dash", orient: str = "landscape") -> dict:
    spec = scene(note, 32, ("gaia", "terrain", "markets", "drive"))
    ctl = controls(orient)
    spec["controls"] = ctl
    spec["protocol"] = "goldend-osai-omega/1"
    spec["launch"] = False
    return spec


def write(out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (Path.home() / ".juniorhome" / "gaia_mesh")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = board()
    path = out_dir / "ue5_controls.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return {"path": str(path), "actors": len(spec.get("actors") or []), "launch": False}
