"""Propose missing title-block fields. Never silent."""
from __future__ import annotations


def propose(holes: list[str], sidecar: str) -> dict[str, str]:
    out: dict[str, str] = {}
    low = sidecar.lower()
    for h in holes:
        if h == "elevation" and "elev" not in low:
            out[h] = "unknown — do not extrude as final"
        if h == "raster_needs_vectorize":
            out[h] = "run local vectorize; keep scan as parent"
        if h == "native_dwg_not_parsed":
            out[h] = "export DXF R12 ascii before ingest"
    return out
