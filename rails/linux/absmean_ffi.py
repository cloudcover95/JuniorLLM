"""Load absmean.c if compiled. Else Python absmean. No numpy required."""
from __future__ import annotations

import ctypes
from pathlib import Path

from junior_bitnet.math import absmean as py_absmean

ROOT = Path(__file__).resolve().parent


def _lib():
    for name in ("libjunior_absmean.so", "libjunior_absmean.dylib"):
        p = ROOT / name
        if p.is_file():
            lib = ctypes.CDLL(str(p))
            lib.junior_absmean.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
            ]
            lib.junior_absmean.restype = ctypes.c_int
            return lib
    return None


def pack(xs: list[float]) -> dict:
    lib = _lib()
    if lib is None:
        t, g = py_absmean(xs)
        return {"trit": t, "gamma": g, "backend": "python"}
    n = len(xs)
    arr = (ctypes.c_float * n)(*[float(x) for x in xs])
    out = (ctypes.c_int8 * n)()
    g = ctypes.c_float()
    rc = lib.junior_absmean(arr, out, ctypes.byref(g), n)
    if rc != 0:
        t, gg = py_absmean(xs)
        return {"trit": t, "gamma": gg, "backend": "python"}
    return {"trit": [int(out[i]) for i in range(n)], "gamma": float(g.value), "backend": "c"}
