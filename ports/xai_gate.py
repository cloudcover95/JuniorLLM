"""xAI HTTP API is pay-per-token. Home stays local unless credits are proven free.
No network. No key in repo.
"""
from __future__ import annotations

import os


def status() -> dict:
    key = bool(os.environ.get("XAI_API_KEY"))
    free_only = os.environ.get("XAI_FREE_ONLY", "1") != "0"
    # Official docs (docs.x.ai pricing): no unlimited free tier.
    # Signup promo / data-share credits live on console.x.ai, not here.
    return {
        "provider": "xai",
        "base": "https://api.x.ai/v1",
        "paid_api": True,
        "unlimited_free": False,
        "key_present": key,
        "free_only": free_only,
        "call": False,
        "reason": "no permanent free API; use Grok app quota or local llama/i2sd",
        "fallback": "i2sd",
    }


def complete(_text: str) -> dict:
    s = status()
    s["ok"] = False
    return s
