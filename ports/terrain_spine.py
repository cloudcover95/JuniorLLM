"""Local Flagstaff prior. Not USGS. Not Cesium. Not a live Open-Elevation call."""
from __future__ import annotations

import json
import math
from pathlib import Path

FLAGSTAFF_M = 2100.0
SPAN_M = 200.0
OSS = {
    "open_elevation": "self-host SRTM JSON — operator box only, no Home docker",
    "nextzen": "AWS Open Data tiles — drop a json grid in ~/.juniorhome/tiles/",
    "usgs_fetch": False,
    "cesium_ion": False,
}
TILE = Path.home() / ".juniorhome" / "tiles" / "flagstaff.json"


def flagstaff_mesh(n: int = 48, seed: int = 42) -> list[list[float]]:
    grid = []
    for i in range(n):
        row = []
        for j in range(n):
            u, v = i / max(1, n - 1), j / max(1, n - 1)
            z = FLAGSTAFF_M + SPAN_M * (u - 0.5) * 0.4
            z += 18.0 * math.sin((u * 7 + seed % 9) * 1.3)
            z += 11.0 * math.cos((v * 5 + (seed >> 2) % 6) * 1.1)
            row.append(z)
        grid.append(row)
    return grid


def load_local_tile(path: Path | None = None, n: int = 48) -> list[list[float]] | None:
    p = path or TILE
    if not p.is_file():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    grid = raw.get("grid") or raw.get("elevations")
    if not isinstance(grid, list) or not grid:
        return None
    return [list(map(float, row)) for row in grid][:n]


def fetch_oss_flagstaff_mesh(n: int = 48) -> list[list[float]]:
    """Gemini name kept. Network never. Local tile wins if present."""
    return load_local_tile(n=n) or flagstaff_mesh(n)
