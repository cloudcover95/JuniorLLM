"""Host alignment. No device alloc."""
from __future__ import annotations

import ctypes


def aligned_f32(xs: list[float], align: int = 16) -> tuple[object, int]:
    n = len(xs)
    raw = ctypes.create_string_buffer(n * 4 + align)
    addr = ctypes.addressof(raw)
    off = (align - (addr % align)) % align
    buf = (ctypes.c_float * n).from_buffer(raw, off)
    for i, x in enumerate(xs):
        buf[i] = float(x)
    return buf, ctypes.addressof(buf)


def report(n: int = 32) -> dict:
    xs = [0.1 * i for i in range(n)]
    _, addr = aligned_f32(xs)
    return {"n": n, "addr": addr, "align16": addr % 16 == 0, "cudaMalloc": False, "download": False}
