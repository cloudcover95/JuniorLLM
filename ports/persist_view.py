"""Persistence view = existing dense tree html + counts."""
from __future__ import annotations

from pathlib import Path

from ports.tree_dense import HTML, TREE, grow


def view() -> dict:
    if not TREE.is_file():
        grow()
    n = 0
    if TREE.is_file():
        n = sum(1 for line in TREE.read_text(encoding="utf-8").splitlines() if line.strip())
    return {
        "nodes": n,
        "jsonl": str(TREE),
        "html": str(HTML),
        "parquet": False,
        "spiffe": False,
        "viz": "svg-spiral",
    }
