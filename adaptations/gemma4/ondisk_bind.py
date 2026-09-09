"""B3 — Gemma4 checkpoint resolution via ports.ondisk. Never fetch."""
from __future__ import annotations

from pathlib import Path

from ports.ondisk import DiskPort, probe

PORT = "JuniorGemma4-4B"
FALLBACK = "JuniorBitNetFieldCore"


def gemma_row(root: Path | None = None) -> DiskPort:
    for row in probe(root):
        if row.name == PORT:
            return row
    return DiskPort(PORT, False, None, FALLBACK)


def resolve_checkpoint(root: Path | None = None) -> tuple[str | None, str]:
    """Return (path or None, active port). Missing weights stay on FieldCore."""
    row = gemma_row(root)
    if row.present and row.path:
        return row.path, PORT
    return None, row.fallback


def notes(root: Path | None = None) -> dict[str, str | bool | None]:
    path, backend = resolve_checkpoint(root)
    return {
        "port": PORT,
        "present": path is not None,
        "path": path,
        "backend": backend,
        "fetch": False,
    }
