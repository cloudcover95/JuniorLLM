"""UE5 scene port. Writes JSON actors. Does not launch Unreal."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ports.gaia_terrain import write as terrain_write


def scene(note: str, n: int = 32) -> dict:
    launch = os.environ.get("JUNIOR_UE5") == "1"
    return {
        "engine": "UE5",
        "launch": False,
        "launch_allowed": launch,
        "reason": "watts-and-env gate; Home os_route.launch stays false",
        "actors": [
            {"class": "StaticMeshActor", "name": "GaiaTerrain", "mesh": f"gaia_terrain_{n}.obj"},
            {"class": "PointLight", "name": "Goldend", "intensity": 8.0},
        ],
        "note": note,
        "download": False,
    }


def write(out_dir: Path, note: str = "gaia terrain", n: int = 32) -> dict:
    t = terrain_write(out_dir, note, n)
    spec = scene(note, n)
    path = out_dir / "gaia_ue5.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    spec["ue5_json"] = str(path)
    spec["obj"] = t.get("obj")
    spec["faces"] = t.get("faces")
    return spec
