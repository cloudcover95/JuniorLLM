"""B2 — look for local GGUF/MLX dirs. Never fetch."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HOME = Path.home() / ".juniorllm" / "models"

CANDIDATES = {
    "BitNet-2B4T": ("bitnet-b1.58-2B-4T-I2_S.gguf",),
    "JuniorGemma4-4B": ("gemma4-4b-q4_k_m.gguf", "gemma4-4b-bitnet-1.58.mlx"),
    "Qwen-local": ("qwen3-8b-q4_k_m.gguf",),
}


@dataclass(frozen=True)
class DiskPort:
    name: str
    present: bool
    path: str | None
    fallback: str


def probe(root: Path | None = None) -> list[DiskPort]:
    base = root or HOME
    out: list[DiskPort] = []
    for name, files in CANDIDATES.items():
        hit = None
        for fn in files:
            p = base / fn
            if p.exists():
                hit = str(p)
                break
        out.append(DiskPort(name, hit is not None, hit, "JuniorBitNetFieldCore"))
    return out


def ready(name: str, root: Path | None = None) -> bool:
    return any(p.name == name and p.present for p in probe(root))
