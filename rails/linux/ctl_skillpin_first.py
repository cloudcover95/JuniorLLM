"""T21 — skill-pin first. Loopback only. Never fetch. Never exec."""
from __future__ import annotations

from pathlib import Path

from rails.linux.ctl_skillpin import _denied, _guard_root


def skill_pin_first(root: str | None = None) -> dict:
    """T21 — report first SKILL_PINS.jsonl row under a root. Never body. Never fetch. Never exec."""
    from junior_aie.skill_pin import ZERO, SkillDenied, SkillPins

    raw, err = _guard_root(root)
    if err:
        return err
    try:
        pins = SkillPins(Path(raw))
        rows = pins._rows()
    except SkillDenied as exc:
        return _denied(str(exc), raw)

    issues: list[str] = []
    chain_ok = pins.verify_chain()
    if not chain_ok:
        issues.append("chain_break")
    first = rows[0] if rows else None
    last = rows[-1] if rows else None
    return {
        "ok": chain_ok,
        "issues": issues,
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "op": "first",
        "tip": pins.tip(),
        "hdr": first.hdr if first else ZERO,
        "height": first.height if first is not None else -1,
        "count": 1 if first is not None else 0,
        "total": len(rows),
        "name": first.name if first else "",
        "rel": first.rel if first else "",
        "last_op": first.op if first else "",
        "sha256": first.sha256 if first else ZERO,
        "size": first.size if first else 0,
        "prev": first.prev if first else ZERO,
        "at": first.at if first else "",
        "found": first is not None,
        "skills": [],
        "chain_ok": chain_ok,
        "empty": last is None,
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
        "exec": False,
    }
