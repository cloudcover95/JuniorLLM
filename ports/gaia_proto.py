"""Validate Gaia / OSai / Omega envelopes. Stdlib only."""
from __future__ import annotations

import os
from pathlib import Path

from ports.gaia_id import status as id_status
from ports.gaia_sys import system

PROTO = "goldend-osai-omega/1"
JOBS = {"dash-viewport", "gaia-spine", "terrain-obj", "agi-capsule"}


def load_proto() -> dict:
    p = Path(__file__).resolve().parents[1] / "config" / "protocol.toml"
    return {"id": PROTO, "jobs": sorted(JOBS), "toml": p.is_file()}


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
    job = (env.get("omega") or {}).get("job")
    if job and job not in JOBS:
        bad.append("job")
    return bad


def handshake(note: str = "home dash", **kw) -> dict:
    job = kw.pop("job", None)
    strict = bool(kw.pop("strict", False) or os.environ.get("JUNIOR_STRICT"))
    env = system(note, **kw)
    if job:
        omega = dict(env.get("omega") or {})
        omega["job"] = job
        omega["launch"] = False
        env["omega"] = omega
    bad = check_system(env)
    ident = id_status(env)
    if not ident.get("verified"):
        bad.append("identity")
    env["protocol"] = PROTO
    env["schema_ok"] = not bad
    env["schema_bad"] = bad
    env["identity"] = ident
    env["jobs_allowed"] = sorted(JOBS)
    if strict and bad:
        raise ValueError("handshake " + ",".join(bad))
    return env
