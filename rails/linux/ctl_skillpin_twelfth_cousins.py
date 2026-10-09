"""T59 — skill-pin twelfth-cousins. Loopback only. Never fetch. Never exec."""
from __future__ import annotations

from pathlib import Path

from rails.linux.ctl_skillpin import _denied, _guard_root

_MISSING = (
    "missing_parent",
    "missing_grandparent",
    "missing_great_grandparent",
    "missing_great_great_grandparent",
    "missing_great_great_great_grandparent",
    "missing_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_great_great_great_great_grandparent",
    "missing_great_great_great_four_great_great_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_great_great_great_great_great_great_grandparent",
)
