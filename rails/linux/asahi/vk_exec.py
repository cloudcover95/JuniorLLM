"""Execute trit.comp semantics on CPU. SPIR-V if tools exist. No probe object."""
from __future__ import annotations

import math
import shutil
import subprocess
import time
from pathlib import Path

from junior_bitnet.math import absmean

COMP = Path(__file__).with_name("trit.comp")


def exec_shader(xs: list[float] | None = None) -> dict:
    xs = xs or [math.sin(i * 0.13) for i in range(256)]
    t0 = time.perf_counter()
    trit, g = absmean(xs)
    ms = (time.perf_counter() - t0) * 1000
    spv = False
    exe = shutil.which("glslangValidator")
    if exe:
        out = COMP.with_suffix(".spv")
        r = subprocess.run([exe, "-V", str(COMP), "-o", str(out)], capture_output=True, text=True)
        spv = r.returncode == 0
    return {
        "shader": str(COMP),
        "equiv": "absmean",
        "n": len(xs),
        "gamma": g,
        "trit_head": trit[:8],
        "ms": round(ms, 3),
        "spirv": spv,
        "vk_submit": False,
        "download": False,
    }
