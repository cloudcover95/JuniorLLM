"""B4 — Qwen-local checkpoint resolution via ports.ondisk. Never fetch."""
from __future__ import annotations

from pathlib import Path

from ports.ondisk import DiskPort, probe

PORT = "Qwen-local"
GEMMA = "JuniorGemma4-4B"
FALLBACK = "JuniorBitNetFieldCore"


def qwen_row(root: Path | None = None) -> DiskPort:
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
    """Return (path or None, active port). Missing Qwen stays on Gemma or FieldCore."""
    row = qwen_row(root)
    if row.present and row.path:
        return row.path, PORT
    gemma = _row(GEMMA, root)
    if gemma is not None and gemma.present and gemma.path:
        return gemma.path, GEMMA
    return None, row.fallback


def notes(root: Path | None = None) -> dict[str, str | bool | None]:
    path, backend = resolve_checkpoint(root)
    return {
        "port": PORT,
        "present": path is not None,
        "path": path,
        "backend": backend,
        "fetch": False,
        "cap_gb": 8,
    }
