"""Tag a ternary IQ state with the MemSys toy SIS commitment if importable."""
from __future__ import annotations

from adaptations.astra_reason.rigid_iq import run_iq


def tag(text: str) -> dict:
    tr = run_iq(text, loops=2)
    payload = {"rec": tr.recommendation, "rigidity": tr.rigidity}
    try:
        from junior_memsys_suite.lattice.msis_commit import commit

        c = commit(tr.state)
        payload["msis_c0"] = c.c[0]
        payload["msis"] = True
    except Exception:
        payload["msis"] = False
    return payload
