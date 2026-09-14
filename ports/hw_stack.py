"""Layer → hardware. Rust crate omitted on purpose."""
from __future__ import annotations

from pathlib import Path

from ports.t_layers import LAYERS, run as t_run
from rails.linux.backend import probe
from rails.linux.vulkan_probe import probe as vk

SO = Path(__file__).resolve().parents[1] / "rails" / "linux"


def stack(note: str = "home dash") -> dict:
    box = probe()
    c_so = (SO / "libjunior_absmean.so").is_file() or (SO / "libjunior_absmean.dylib").is_file()
    return {
        "layers": LAYERS,
        "t0": t_run("T0", note),
        "box": {"accel": box.accel, "asahi": box.asahi, "machine": box.machine},
        "c_so": c_so,
        "simd": "neon|sse2 if compiled",
        "rust_ffi": False,
        "vulkan": vk(),
        "protocol": "goldend-osai-omega/1",
    }
