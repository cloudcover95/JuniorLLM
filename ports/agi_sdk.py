"""AGI_SDK port card. Jobs only. No weight pull."""
from __future__ import annotations

JOBS = ("dash-viewport", "gaia-spine", "terrain-obj", "agi-capsule")


def card() -> dict:
    return {
        "name": "AGI_SDK",
        "jobs": list(JOBS),
        "download_gb": 0.0,
        "ue5_launch": False,
        "model_pull": False,
    }
