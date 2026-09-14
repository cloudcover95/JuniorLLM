"""Time Flagstaff + winsor + handshake. Stdlib."""
from __future__ import annotations

import time

from ports.agent_pipe import step
from ports.flagstaff_balance import check
from ports.gaia_proto import handshake
from rails.linux.absmean_ffi import pack as cpack


def bench(note: str = "buy oats", n: int = 20) -> dict:
    t0 = time.perf_counter()
    for _ in range(n):
        check(note)
    t_vote = (time.perf_counter() - t0) * 1000 / n
    t0 = time.perf_counter()
    env = None
    for _ in range(n):
        env = handshake(note, job="dash-viewport")
    t_hs = (time.perf_counter() - t0) * 1000 / n
    xs = [float((ord(c) % 13) - 6) for c in note] or [0.2]
    t0 = time.perf_counter()
    c = None
    for _ in range(n):
        c = cpack(xs)
    t_c = (time.perf_counter() - t0) * 1000 / n
    row = step(note, write=False)
    return {
        "n": n,
        "ms_vote": round(t_vote, 3),
        "ms_handshake": round(t_hs, 3),
        "ms_c_or_py": round(t_c, 3),
        "c_backend": (c or {}).get("backend"),
        "schema_ok": (env or {}).get("schema_ok"),
        "votes_ok": row.get("ok"),
        "profit": row.get("profit"),
        "train": False,
        "web3node": "pointer",
        "download": False,
    }
