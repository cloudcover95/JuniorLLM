"""Compile rails/linux/i2s_pack.c when gcc exists."""
from __future__ import annotations

import ctypes
import subprocess
import tempfile
from pathlib import Path

from junior_bitnet.i2s import pack as py_pack

SRC = Path(__file__).resolve().parents[1] / "rails" / "linux" / "i2s_pack.c"


def build(dest: Path | None = None) -> Path | None:
    dest = dest or Path(tempfile.gettempdir()) / "libi2s_pack.so"
    r = subprocess.run(["gcc", "-O2", "-fPIC", "-shared", str(SRC), "-o", str(dest)], capture_output=True)
    return dest if r.returncode == 0 and dest.is_file() else None


def pack_c(trits: list[int], so: Path) -> bytes:
    lib = ctypes.CDLL(str(so))
    lib.i2s_pack.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte)]
    lib.i2s_pack.restype = ctypes.c_int
    n = len(trits)
    arr = (ctypes.c_int * n)(*trits)
    out = (ctypes.c_ubyte * (n + 4))()
    k = lib.i2s_pack(arr, n, out)
    return bytes(out[:k])


def matches(trits: list[int]) -> bool:
    so = build()
    if so is None:
        return False
    return pack_c(trits, so) == py_pack(trits)
