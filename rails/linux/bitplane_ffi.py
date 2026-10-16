"""FFI for junior_bitplane. Local .so only. No LUT."""
from __future__ import annotations
import ctypes
from pathlib import Path
import numpy as np

def _load():
    p = Path(__file__).with_name("libjunior_bitplane.so")
    if not p.is_file():
        return None
    lib = ctypes.CDLL(str(p))
    lib.junior_bitplane.argtypes = [
        ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_size_t, ctypes.c_size_t,
    ]
    lib.junior_bitplane.restype = ctypes.c_int
    return lib

def bitplane(T: np.ndarray, x: np.ndarray, gamma: float = 1.0) -> np.ndarray | None:
    lib = _load()
    if lib is None:
        return None
    T = np.ascontiguousarray(T, dtype=np.int8)
    x = np.ascontiguousarray(x, dtype=np.float32)
    m, k = T.shape
    y = np.empty(m, dtype=np.float32)
    rc = lib.junior_bitplane(
        T.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(gamma), m, k,
    )
    return y if rc == 0 else None
