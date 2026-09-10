"""B2 — look for local GGUF/MLX dirs. Never fetch."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HOME = Path.home() / ".juniorllm" / "models"

CANDIDATES = {
    "BitNet-2B4T": ("bitnet-b1.58-2B-4T-I2_S.gguf",),
    "JuniorGemma4-4B": ("gemma4-4b-q4_k_m.gguf", "gemma4-4b-bitnet-1.58.mlx"),
    "Qwen-local": ("qwen3-8b-q4_k_m.gguf",),
    "JuniorKimiK3-edge": (
        "kimi-k3-edge-pruned-8gb.mlx",
        "kimi-k3-edge-pruned.gguf",
    ),
}

CAP_BYTES = 8 * 1024 * 1024 * 1024
FULL_WEIGHT_MARKERS = (
    "1.5tb",
    "1.45tb",
    "kimi-k3-full",
    "moonshot-kimi-k3",
)


@dataclass(frozen=True)
class DiskPort:
    name: str
    present: bool
    path: str | None
    fallback: str


def _blocked(name: str, path: Path) -> bool:
    blob = name.lower() + " " + path.name.lower()
    if any(m in blob for m in FULL_WEIGHT_MARKERS):
        return True
    try:
        if path.exists() and path.stat().st_size > CAP_BYTES:
            return True
    except OSError:
        return True
    return False


def probe(root: Path | None = None) -> list[DiskPort]:
    base = root or HOME
    out: list[DiskPort] = []
    for name, files in CANDIDATES.items():
        hit = None
        for fn in files:
            p = base / fn
            if p.exists() and not _blocked(name, p):
                hit = str(p)
                break
        out.append(DiskPort(name, hit is not None, hit, "JuniorBitNetFieldCore"))
    return out


def ready(name: str, root: Path | None = None) -> bool:
    return any(p.name == name and p.present for p in probe(root))
