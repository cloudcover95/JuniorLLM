"""Accel probe for JuniorOS overlay. Same ternary math on every arch.

mlx   — Apple Silicon when mlx imports
cuda  — NVIDIA when torch.cuda is available
cpu   — aarch64 / x86_64 / armv7 fallback (stdlib AbsMean)
"""
from __future__ import annotations

import platform
from dataclasses import asdict, dataclass

from junior_bitnet.math import absmean


@dataclass(frozen=True)
class Probe:
    machine: str
    system: str
    accel: str
    kernel: str
    notes: str


def _accel() -> tuple[str, str]:
    try:
        import mlx.core as mx  # noqa: F401

        return "mlx", "BitNet-mlx when on PYTHONPATH"
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name(0)
    except Exception:
        pass
    mach = platform.machine().lower()
    if mach in {"aarch64", "arm64", "armv7l", "armv8l"}:
        return "cpu", "ARM stdlib AbsMean"
    return "cpu", "x86 stdlib AbsMean"


def probe() -> Probe:
    accel, notes = _accel()
    return Probe(platform.machine(), platform.system(), accel, "junior_bitnet.absmean", notes)


def trit(xs: list[float]) -> list[int]:
    z, _ = absmean(xs)
    return z


def report() -> dict:
    p = probe()
    z = trit([40.0, 20.0, 8.0, 0.2, -3.0])
    return {**asdict(p), "sample_trits": z, "alphabet": all(t in (-1, 0, 1) for t in z)}
