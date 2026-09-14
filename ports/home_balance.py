"""Home suite balance. Honeykrisp is a host driver, not a dispatch."""
from __future__ import annotations

from ports.flagstaff_balance import check
from ports.junioros import boot
from ports.ship_trit import ship
from ports.xyz_check import xyz


def balance(note: str = "home dash") -> dict:
    gate = check(note)
    return {
        "ok": gate.get("ok"),
        "votes": gate.get("votes"),
        "area": gate.get("area"),
        "os": boot(note),
        "trit": ship(note),
        "xyz": {"X": xyz(note).get("X_junior"), "profit": (xyz(note).get("X_junior") or {}).get("profit")},
        "honeykrisp": {"role": "asahi-vulkan-host", "submit": False},
        "domains": ["loopback", "cpu-trit", "optional-c-so", "fieldcore-jsonl"],
        "download": False,
    }
