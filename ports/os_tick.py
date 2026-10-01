"""JuniorOS tick. Uses the host envelope unless named."""
from __future__ import annotations

import time

from ports.osai_envelope import run


def tick(task: str = "JuniorOS", env: str = "t4") -> dict:
    t0 = time.perf_counter()
    row = run(task, env)
    row["ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return row
