"""Guess height only from explicit tokens. No silent extrusion."""
from __future__ import annotations

import re


def guess_height(text: str) -> float | None:
    m = re.search(r"THICK(?:NESS)?\s*[:#]?\s*([0-9.]+)", text, re.I)
    if m:
        return float(m.group(1))
    return None
