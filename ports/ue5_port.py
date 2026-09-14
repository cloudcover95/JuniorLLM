"""UE5 scene port. JSON only. launch stays false unless operator sets JUNIOR_UE5=1 *and* calls launch()."""
from __future__ import annotations

import json
import os
from pathlib import Path

SURFACES = ("gaia", "terrain", "markets", "dash", "drive")


def scene(note: str, n: int = 32, surfaces: tuple[str, ...] = ("gaia", "terrain")) -> dict:
    allowed = os.environ.get("JUNIOR_UE5") == "1"
    actors = [{"class": "CameraActor", "name": "DashCam", "orient": "landscape"}]
    if "terrain" in surfaces or "gaia" in surfaces:
        actors.append({"class": "StaticMeshActor", "name": "GaiaTerrain", "mesh": f"gaia_terrain_{n}.obj"})
        actors.append({"class": "PointLight", "name": "Goldend", "intensity": 8.0})
    if "markets" in surfaces:
        actors.append({"class": "TextRenderActor", "name": "Tape", "source": "local-seed"})
    if "drive" in surfaces:
        actors.append({"class": "Pawn", "name": "DriveRig", "sim": True})
    return {
        "engine": "UE5",
        "protocol": "goldend-osai-omega/1",
        "job": "dash-viewport",
        "launch": False,
        "launch_allowed": allowed,
        "surfaces": [s for s in surfaces if s in SURFACES],
        "actors": actors,
        "note": note,
        "download": False,
    }


def launch() -> dict:
    """Never called by Home automations."""
    return {"launch": False, "reason": "os_route gate"}


def write(out_dir: Path, note: str = "gaia terrain", n: int = 32, surfaces: tuple[str, ...] = ("gaia", "terrain")) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = scene(note, n, surfaces)
    path = out_dir / "gaia_ue5.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    spec["ue5_json"] = str(path)
    return spec
