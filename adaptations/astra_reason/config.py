"""Astra-class stand-in config. High-quant base + BitNet loops."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AstraReasonConfig:
    # High-quant bases we are allowed to point at (not closed Astra).
    gemma_id: str = "google/gemma-4-E4B-it"
    qwen_id: str = "Qwen/Qwen3-8B"  # local GGUF expected; do not auto-download
    family: str = "gemma"  # gemma | qwen
    quant: str = "Q4_K_M"
    max_download_gb: float = 8.0
    loops: int = 4  # rigid IQ depth (reuse ternary stack)
    dim: int = 32
    mem_slots: int = 64
    backend: str = "mlx-or-gguf"

    def model_id(self) -> str:
        return self.gemma_id if self.family == "gemma" else self.qwen_id
