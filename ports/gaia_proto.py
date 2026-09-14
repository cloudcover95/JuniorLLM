"""Validate Gaia / OSai / Omega envelopes. Stdlib only."""
from __future__ import annotations

from ports.gaia_sys import system


def check_system(env: dict) -> list[str]:
    bad = []
    if env.get("system") != "JuniorGaia":
        bad.append("system")
    if env.get("ue5_launch") is not False:
        bad.append("ue5_launch")
    if env.get("download") is not False:
        bad.append("download")
    who = env.get("who") or {}
    if not who.get("name") or not who.get("pronouns"):
        bad.append("who")
    view = env.get("view") or {}
    if view.get("orient") not in ("portrait", "landscape"):
        bad.append("orient")
    note = env.get("note") or {}
    if not note.get("port") or not str(note.get("port", "")).startswith("Junior"):
        bad.append("port")
    return bad


def handshake(note: str = "home dash", **kw) -> dict:
    env = system(note, **kw)
    bad = check_system(env)
    env["protocol"] = "goldend-osai-omega/1"
    env["schema_ok"] = not bad
    env["schema_bad"] = bad
    return env
