"""Where free credits *could* sit. Keys stay in env. No HTTP from this module."""
from __future__ import annotations

import os

from ports.xai_gate import status as xai_status

BOARD = (
    {"id": "local", "env": None, "card": False, "signup": False, "train": False, "note": "i2sd / llama.cpp"},
    {"id": "xai", "env": "XAI_API_KEY", "card": True, "signup": True, "train": True, "note": "paid API; promo credits only"},
    {"id": "groq", "env": "GROQ_API_KEY", "card": False, "signup": True, "train": False, "note": "free tier gpt-oss/qwen 30rpm 1k rpd"},
    {"id": "gemini", "env": "GEMINI_API_KEY", "card": False, "signup": True, "train": True, "note": "AI Studio free; trains on free tier"},
    {"id": "openrouter", "env": "OPENROUTER_API_KEY", "card": False, "signup": True, "train": True, "note": ":free models 50 rpd"},
    {"id": "cloudflare", "env": "CF_API_TOKEN", "card": False, "signup": True, "train": False, "note": "Workers AI neurons/day"},
    {"id": "mistral", "env": "MISTRAL_API_KEY", "card": False, "signup": True, "train": False, "note": "free-plan credits"},
)


def board() -> list[dict]:
    out = []
    for row in BOARD:
        key = row["env"]
        item = dict(row)
        item["key_present"] = bool(key and os.environ.get(key))
        item["call"] = False
        out.append(item)
    x = xai_status()
    out[1]["xai"] = {"unlimited_free": x["unlimited_free"], "fallback": x["fallback"]}
    return out


def ready() -> list[str]:
    return [r["id"] for r in board() if r["id"] == "local" or r["key_present"]]
