"""Home imager. Trit mesh default. SVD opt-in. No parquet. No FAISS theater."""
from __future__ import annotations

import json
import time
from pathlib import Path

from ports.fieldcore_spine import expand
from ports.flagstaff_balance import check
from ports.gaia_proto import handshake

OUT = Path.home() / ".juniorhome" / "gaia_mesh" / "imager.jsonl"


def image(note: str = "journal field note", n: int = 32, *, full_svd: bool = False) -> dict:
    gate = check(note)
    hs = handshake(note, job="dash-viewport") if gate.get("ok") else {}
    t0 = time.perf_counter()
    fc = expand(n=n, k=8, full_svd=full_svd) if gate.get("ok") else {"skipped": True}
    ms = (time.perf_counter() - t0) * 1000
    row = {
        "ok": gate.get("ok"),
        "backend": fc.get("backend", "skipped"),
        "n": fc.get("n"),
        "gamma": fc.get("gamma") or (hs.get("note") or {}).get("gamma"),
        "sparsity": fc.get("sparsity"),
        "ms": round(ms, 3),
        "svd": bool(fc.get("svd")),
        "parquet": False,
        "faiss_fake": False,
        "mlx_required": False,
        "zone": str(OUT.parent),
        "protocol": hs.get("protocol") or "goldend-osai-omega/1",
    }
    if gate.get("ok"):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.open("a", encoding="utf-8").write(json.dumps(row) + "\n")
        row["wrote"] = True
    return row
