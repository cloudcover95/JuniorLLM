"""Validate Gaia / OSai / Omega envelopes. Stdlib only."""
from __future__ import annotations

from pathlib import Path

from ports.gaia_sys import system

PROTO = "goldend-osai-omega/1"
JOBS = {"dash-viewport", "gaia-spine", "terrain-obj", "agi-capsule"}


def load_proto() -> dict:
    p = Path(__file__).resolve().parents[1] / "config" / "protocol.toml"
    if not p.is_file():
        return {"id": PROTO, "jobs": sorted(JOBS)}
    text = p.read_text(encoding="utf-8")
    jobs = [ln.split("\"")[1] for ln in text.splitlines() if "dash-viewport" in ln or "gaia-spine" in ln or "terrain-obj" in ln or "agi-capsule" in ln]
    return {"id": PROTO, "jobs": jobs or sorted(JOBS)}


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
    env = system(note, **kw)
    if job:
        omega = dict(env.get("omega") or {})
        omega["job"] = job
        omega["launch"] = False
        env["omega"] = omega
    bad = check_system(env)
    if job and job not in JOBS:
        bad.append("job")
    env["protocol"] = PROTO
    env["schema_ok"] = not bad
    env["schema_bad"] = bad
    env["jobs_allowed"] = sorted(JOBS)
    return env
