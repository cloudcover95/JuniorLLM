"""CPU stand-in for trit.comp. Compiles SPIR-V only if glslangValidator exists."""
from __future__ import annotations

import math
import shutil
import subprocess
import time
from pathlib import Path

from junior_bitnet.math import absmean
from rails.linux.vulkan_probe import probe

COMP = Path(__file__).with_name("trit.comp")


def cpu_dispatch(xs: list[float]) -> dict:
    t0 = time.perf_counter()
    t, g = absmean(xs)
    return {"backend": "cpu-sandbox", "n": len(xs), "gamma": g, "trit": t[:8], "ms": round((time.perf_counter() - t0) * 1000, 3)}


def spirv() -> dict:
    exe = shutil.which("glslangValidator")
    if not exe:
        return {"spirv": False, "reason": "no glslangValidator"}
    out = COMP.with_suffix(".spv")
    r = subprocess.run([exe, "-V", str(COMP), "-o", str(out)], capture_output=True, text=True)
    return {"spirv": r.returncode == 0, "spv": str(out) if r.returncode == 0 else None, "stderr": (r.stderr or "")[:200]}


def run(n: int = 256) -> dict:
    xs = [math.sin(i * 0.13) for i in range(n)]
    return {**probe(), **cpu_dispatch(xs), **spirv(), "asahi_dispatch": False}
