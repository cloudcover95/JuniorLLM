"""FieldCore spine. Live path is trit energy. Full SVD is a probe."""
from __future__ import annotations

import json
import os
from pathlib import Path

from junior_bitnet.winsor import pack
from ports.terrain_spine import flagstaff_mesh


def expand(n: int = 48, k: int = 30, *, full_svd: bool = False, out: Path | None = None) -> dict:
    mesh = flagstaff_mesh(n)
    flat = [z for row in mesh for z in row]
    q = pack(flat)
    row = {
        "n": n,
        "k": k,
        "backend": "trit-energy",
        "gamma": q.get("gamma"),
        "sparsity": q.get("sparsity"),
        "sync": "local-only",
        "cluster": False,
        "svd": False,
    }
    if full_svd or os.environ.get("JUNIOR_SVD") == "1":
        from ports.svd_tick import tick

        svd = tick(n, k, mesh)
        row.update({"svd": True, "energy": svd.get("energy"), "ms": svd.get("ms"), "backend": svd.get("backend")})
    out = out or (Path.home() / ".juniorhome" / "gaia_mesh" / "fieldcore_spine.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return {**row, "path": str(out)}
