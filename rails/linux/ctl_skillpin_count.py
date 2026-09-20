"""T25 — skill-pin count. Loopback only. Never fetch. Never exec."""
from __future__ import annotations

from pathlib import Path

from rails.linux.ctl_skillpin import _denied, _guard_root


def skill_pin_count(root: str | None = None) -> dict:
    """T25 — count SKILL_PINS.jsonl rows under a root. Never body. Never fetch. Never exec."""
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
    last = rows[-1] if rows else None
    first = rows[0] if rows else None
    return {
        "ok": chain_ok,
        "issues": issues,
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "op": "count",
        "tip": pins.tip(),
        "hdr": last.hdr if last else ZERO,
        "height": last.height if last is not None else -1,
        "count": len(rows),
        "total": len(rows),
        "name": last.name if last else "",
        "rel": last.rel if last else "",
        "last_op": last.op if last else "",
        "sha256": last.sha256 if last else ZERO,
        "size": last.size if last else 0,
        "prev": last.prev if last else ZERO,
        "at": last.at if last else "",
        "found": last is not None,
        "skills": [],
        "chain_ok": chain_ok,
        "empty": last is None,
        "genesis": first.hdr if first else ZERO,
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
        "exec": False,
    }
