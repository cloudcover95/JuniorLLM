"""OG imager clone. Separate from ports.imager. Still no parquet FAISS."""
from __future__ import annotations

from ports.imager import image as _live


def image_og(note: str = "home vault memory") -> dict:
    row = _live(note, n=32, full_svd=False)
    row["clone"] = "og"
    row["suite"] = "palace-jsonl"
    return row
