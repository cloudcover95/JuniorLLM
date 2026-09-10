"""B5 — JuniorKimiK3-edge checkpoint resolution via ports.ondisk. Never fetch."""
from __future__ import annotations

from pathlib import Path

from ports.ondisk import DiskPort, probe

PORT = "JuniorKimiK3-edge"
GEMMA = "JuniorGemma4-4B"
FALLBACK = "JuniorBitNetFieldCore"


def kimi_row(root: Path | None = None) -> DiskPort:
    for row in probe(root):
        if row.name == PORT:
            return row
    return DiskPort(PORT, False, None, FALLBACK)


def _row(name: str, root: Path | None = None) -> DiskPort | None:
    for row in probe(root):
        if row.name == name:
            return row
    return None


def resolve_checkpoint(root: Path | None = None) -> tuple[str | None, str]:
    """Return (path or None, active port). Missing edge prune stays on Gemma or FieldCore."""
    row = kimi_row(root)
    if row.present and row.path:
        return row.path, PORT
    gemma = _row(GEMMA, root)
    if gemma is not None and gemma.present and gemma.path:
        return gemma.path, GEMMA
    return None, row.fallback


def notes(root: Path | None = None) -> dict[str, str | bool | None | float]:
    path, backend = resolve_checkpoint(root)
    return {
        "port": PORT,
        "present": path is not None and backend == PORT,
        "path": path if backend == PORT else None,
        "backend": backend,
        "fetch": False,
        "cap_gb": 8,
        "full_kimi_1_5tb": False,
    }
