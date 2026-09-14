"""FieldCore spine. SVD retain-k already in svd_tick. No parquet. No cluster."""
from __future__ import annotations

import json
from pathlib import Path

from ports.svd_tick import tick
from ports.terrain_spine import flagstaff_mesh


def expand(n: int = 48, k: int = 30, out: Path | None = None) -> dict:
    mesh = flagstaff_mesh(n)
    svd = tick(n, k, mesh)
    out = out or (Path.home() / ".juniorhome" / "gaia_mesh" / "fieldcore_spine.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "n": svd.get("n"),
        "k": svd.get("k"),
        "energy": svd.get("energy"),
        "backend": svd.get("backend"),
        "ms": svd.get("ms"),
        "sync": "local-only",
        "cluster": False,
        "parquet": False,
        "mlx": False,
    }
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return {**row, "path": str(out)}
