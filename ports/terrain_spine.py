"""Local Flagstaff prior. Not USGS. Not Cesium. Not a live Open-Elevation call."""
from __future__ import annotations

import math

FLAGSTAFF_M = 2100.0
SPAN_M = 200.0
OSS = {
    "open_elevation": "self-host SRTM JSON — operator box only",
    "nextzen": "AWS Open Data terrain tiles — offline GeoTIFF later",
    "usgs_fetch": False,
    "cesium_ion": False,
}


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


def fetch_oss_flagstaff_mesh(n: int = 48) -> list[list[float]]:
    """Name kept from the Gemini sketch. Does not hit a network."""
    return flagstaff_mesh(n)
