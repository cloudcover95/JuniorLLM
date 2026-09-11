"""Accel probe. Same AbsMean. Detect Asahi / bitnet.cpp binary. No firmware writes."""
from __future__ import annotations

import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from junior_bitnet.math import absmean


@dataclass(frozen=True)
class Probe:
    machine: str
    system: str
    accel: str
    kernel: str
    notes: str
    asahi: bool
    bitnet_cpp: str | None


def _asahi() -> bool:
    for p in ("/etc/os-release", "/usr/lib/os-release"):
        try:
            txt = Path(p).read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        if "asahi" in txt:
            return True
    return False


def _cpp() -> str | None:
    env = os.environ.get("JUNIOR_BITNET_CPP")
    if env and Path(env).is_file():
        return env
    return shutil.which("bitnet-cpp") or shutil.which("llama-cli")


def _accel() -> tuple[str, str]:
    if _asahi():
        try:
            import mlx.core as mx  # noqa: F401

            return "mlx", "Asahi + MLX userspace"
        except Exception:
            return "cpu", "Asahi, MLX not imported"
    try:
        import mlx.core as mx  # noqa: F401

        return "mlx", "BitNet-mlx / Darwin"
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
    return Probe(
        platform.machine(),
        platform.system(),
        accel,
        "junior_bitnet.absmean",
        notes,
        _asahi(),
        _cpp(),
    )


def trit(xs: list[float]) -> list[int]:
    z, _ = absmean(xs)
    return z


def report() -> dict:
    p = probe()
    z = trit([40.0, 20.0, 8.0, 0.2, -3.0])
    return {**asdict(p), "sample_trits": z, "alphabet": all(t in (-1, 0, 1) for t in z)}
