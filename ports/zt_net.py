"""Zero-trust net for Home. Loopback. No implied overlay."""
from __future__ import annotations

from ports.cache_secure import put
from ports.zero_trust import BOUNDS, metric

PRINCIPLES = [
    "never trust the note; Flagstaff AND first",
    "bind 127.0.0.1 only",
    "no download, no rpc, no ue5 launch",
    "trit is not a credential",
    "sha3 identity; I2_S is a neighbor key",
    "colliding packs do not merge receipts",
    "jsonl local; no required cloud IdP",
]


def net(note: str = "home dash") -> dict:
    z = metric(note)
    c = put(note) if z.get("ok") else {"ok": False, "skipped": True}
    return {
        "ok": bool(z.get("ok") and c.get("ok")),
        "principles": PRINCIPLES,
        "bounds": BOUNDS,
        "zero_trust": z.get("net"),
        "collision": c.get("collision"),
        "overlay": False,
        "idp": False,
    }
