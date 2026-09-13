"""JuniorOSai port card. Not a weight download."""
from __future__ import annotations
from ports.registry import LLMPort

JUNIOROSAI = LLMPort(
    "JuniorOSai",
    "bitnet-native",
    "ternary-1.58",
    "numpy-or-mlx",
    0.0,
    "JuniorOS home kernel + Flagstaff budgets. Ship = card+kernel; 2B GGUF optional if on disk.",
)

SIZING = {
    "T0": {"params": 0, "download_gb": 0.0, "quant": "ternary-1.58", "ram_gb": 2},
    "T0_gguf": {"params": 2_000_000_000, "download_gb": 1.5, "quant": "I2_S", "file": "bitnet-b1.58-2B-4T"},
    "T1": {"params": 4_000_000_000, "download_gb": 4.0, "quant": "Q4_K_M", "ram_gb": 24},
}
