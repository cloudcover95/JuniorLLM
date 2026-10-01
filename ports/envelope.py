"""Variable compute envelopes. Caps, not a wattmeter."""
from __future__ import annotations

ENVELOPES = {
    "t4": {"watts": 12, "notes": 8, "chars": 256, "goldens": 4, "download_gb": 0.0, "files": 4},
    "host": {"watts": 45, "notes": 16, "chars": 1024, "goldens": 16, "download_gb": 0.0, "files": 8},
    "operator": {"watts": 45, "notes": 34, "chars": 2048, "goldens": 32, "download_gb": 1.5, "files": 8},
}


def select(name: str = "host") -> dict:
    key = name if name in ENVELOPES else "host"
    row = dict(ENVELOPES[key])
    row["name"] = key
    row["bind"] = "127.0.0.1"
    row["model_pull"] = False
    row["measured"] = False
    return row


def fit(text: str, cap: int) -> str:
    t = text or ""
    return t if len(t) <= cap else t[: cap - 1] + "\u2026"


def slice_rows(rows: list, n: int) -> list:
    return list(rows)[: max(0, n)]
