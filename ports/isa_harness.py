"""What compilers exist. Graphs/tensor cores stay off until threshold."""
from __future__ import annotations

import shutil
import platform

from rails.linux.cuda_graphs import report as cuda_r


def harness() -> dict:
    mach = platform.machine().lower()
    return {
        "machine": mach,
        "cc": bool(shutil.which("cc")),
        "clang": bool(shutil.which("clang")),
        "nvcc": bool(shutil.which("nvcc")),
        "glslang": bool(shutil.which("glslangValidator")),
        "rustc": bool(shutil.which("rustc")),
        "neon": mach in {"aarch64", "arm64"},
        "sse2": mach in {"x86_64", "amd64"},
        "cuda": cuda_r(),
        "use_graph": False,
        "use_wmma": False,
        "osai": "compile host .so if cc; skip rustc/nvcc for T0",
    }
