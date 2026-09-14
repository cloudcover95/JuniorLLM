"""Local identity. Not cloudcover95/edge-architect. Not a token."""
from __future__ import annotations

from ports.gaia import load_who


def status(env: dict | None = None) -> dict:
    who = (env or {}).get("who") or load_who()
    named = bool(who.get("name")) and bool(who.get("pronouns"))
    loop = True  # envelope never binds 0.0.0.0 here
    return {
        "issuer": "local-gaia.json",
        "name": who.get("name"),
        "pronouns": who.get("pronouns"),
        "named": named,
        "loopback": loop,
        "download": False,
        "ue5_launch": False,
        "verified": bool(named and loop),
        "hardcoded_operator": False,
    }
