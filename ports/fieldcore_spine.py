"""FieldCore spine. Trit energy. Cap grid. SVD opt-in."""
from __future__ import annotations

import json
import os
from pathlib import Path

from junior_bitnet.winsor import pack
from ports.terrain_spine import flagstaff_mesh

MAX_N = 64


def expand(n: int = 32, k: int = 8, *, full_svd: bool = False, out: Path | None = None) -> dict:
    n = max(8, min(int(n or 32), MAX_N))
    k = max(2, min(int(k or 8), n))
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
        "max_n": MAX_N,
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
