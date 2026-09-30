"""juniorctl pin/load helpers. Loopback only. Never fetch. Never exec."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path() -> None:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _skill_pin_denied(reason: str, raw: str) -> dict:
    return {
        "ok": False,
        "issues": [reason],
        "bind": "127.0.0.1:8765",
        "root": raw,
        "skills": [],
        "count": 0,
        "chain_ok": False,
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
    }


def skill_pin_pin(rel: str | None = None, root: str | None = None) -> dict:
    """T8 — pin a SKILL.md under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from junior_aie.skill_pin import DENY_FRAGMENTS, SkillDenied, SkillPins

    raw = (root or "").strip() or str(ROOT / "grok_bot")
    low = raw.replace("\\", "/").lower()
    wildcard = ".".join(("0", "0", "0", "0"))
    if any(frag in low for frag in DENY_FRAGMENTS):
        return _skill_pin_denied("denied_name", raw)
    if wildcard in low:
        return _skill_pin_denied("wildcard", raw)
    if ".." in Path(raw).parts:
        return _skill_pin_denied("path_escape", raw)
    text = (rel or "").strip()
    if not text:
        return _skill_pin_denied("empty_path", raw)
    rel_low = text.replace("\\", "/").lower()
    if any(frag in rel_low for frag in DENY_FRAGMENTS):
        return _skill_pin_denied("denied_name", raw)
    if wildcard in rel_low:
        return _skill_pin_denied("wildcard", raw)
    if ".." in Path(text).parts:
        return _skill_pin_denied("path_escape", raw)
    try:
        pins = SkillPins(Path(raw))
        row = pins.pin(text)
    except SkillDenied as exc:
        return _skill_pin_denied(str(exc), raw)
    return {
        "ok": True,
        "issues": [],
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "rel": text,
        "name": row.name,
        "op": row.op,
        "sha256": row.sha256,
        "size": row.size,
        "height": row.height,
        "skills": [],
        "count": 1,
        "chain_ok": pins.verify_chain(),
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
        "exec": False,
    }


def skill_pin_load(rel: str | None = None, root: str | None = None) -> dict:
    """T7 — load a pinned SKILL.md under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from junior_aie.skill_pin import DENY_FRAGMENTS, SkillDenied, SkillPins

    raw = (root or "").strip() or str(ROOT / "grok_bot")
    low = raw.replace("\\", "/").lower()
    wildcard = ".".join(("0", "0", "0", "0"))
    if any(frag in low for frag in DENY_FRAGMENTS):
        return _skill_pin_denied("denied_name", raw)
    if wildcard in low:
        return _skill_pin_denied("wildcard", raw)
    if ".." in Path(raw).parts:
        return _skill_pin_denied("path_escape", raw)
    text = (rel or "").strip()
    if not text:
        return _skill_pin_denied("empty_path", raw)
    rel_low = text.replace("\\", "/").lower()
    if any(frag in rel_low for frag in DENY_FRAGMENTS):
        return _skill_pin_denied("denied_name", raw)
    if wildcard in rel_low:
        return _skill_pin_denied("wildcard", raw)
    if ".." in Path(text).parts:
        return _skill_pin_denied("path_escape", raw)
    try:
        pins = SkillPins(Path(raw))
        body, row = pins.load(text)
    except SkillDenied as exc:
        return _skill_pin_denied(str(exc), raw)
    return {
        "ok": True,
        "issues": [],
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "rel": text,
        "name": row.name,
        "op": row.op,
        "sha256": row.sha256,
        "size": row.size,
        "height": row.height,
        "body": body,
        "skills": [],
        "count": 1,
        "chain_ok": pins.verify_chain(),
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
        "exec": False,
    }
