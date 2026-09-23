#!/usr/bin/env python3
"""juniorctl — JuniorOS entry. skill_pin_list skill_pin_verify skill_pin_verify_one skill_pin_tip skill_pin_log skill_pin_height skill_pin_get skill_pin_at skill_pin_range skill_pin_since skill_pin_until skill_pin_before skill_pin_after skill_pin_first skill_pin_last skill_pin_tail skill_pin_head skill_pin_count skill_pin_genesis skill_pin_parent skill_pin_child skill_pin_children skill_pin_ancestors skill_pin_siblings"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINUX = Path(__file__).resolve().parent

HARDENING = (
    "NoNewPrivileges=yes",
    "ProtectSystem=strict",
    "MemoryDenyWriteExecute=yes",
    "CapabilityBoundingSet=",
    "JUNIOR_BIND=127.0.0.1:8765",
)


def _path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
