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
    if "view" not in low:
        out.setdefault("view_links", "add VIEW: plan,elev")
    if "dim_ok" not in low:
        out.setdefault("dim_ok", "add DIM_OK: yes only after tape-check")
    return out
