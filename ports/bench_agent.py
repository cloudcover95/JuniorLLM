"""Time Flagstaff + handshake + xyz. Local only."""
from __future__ import annotations

import time

from ports.agent_pipe import step
from ports.osai_suite import run_all

NOTES = ("buy oats", "cad title block", "bind 0.0.0.0", "porch light 127.0.0.1")


def bench() -> dict:
    rows = []
    t0 = time.perf_counter()
    for n in NOTES:
        s = time.perf_counter()
        r = step(n, write=False)
        rows.append({"note": n, "ok": r.get("ok"), "ms": round((time.perf_counter() - s) * 1000, 3), "profit": r.get("profit")})
    suite = run_all()
    return {
        "rows": rows,
        "pipe_ms": round((time.perf_counter() - t0) * 1000, 3),
        "osai_passed": suite.get("passed"),
        "osai_n": suite.get("n"),
        "train": False,
        "web3node": "pointer",
        "download": False,
    }
