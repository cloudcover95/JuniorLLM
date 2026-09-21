"""T26 — skill-pin genesis. Loopback only. Never fetch. Never exec."""
from __future__ import annotations

from pathlib import Path

from rails.linux.ctl_skillpin import _denied, _guard_root


def skill_pin_genesis(root: str | None = None) -> dict:
    """T26 — report genesis SKILL_PINS.jsonl row under a root. Never body. Never fetch. Never exec."""
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
    genesis_hdr = first.hdr if first else ZERO
    prev = first.prev if first else ZERO
    if first is not None and prev != ZERO:
        issues.append("genesis_prev")
        chain_ok = False
    if first is not None and first.height != 0:
        issues.append("genesis_height")
        chain_ok = False
    return {
        "ok": chain_ok,
        "issues": issues,
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "op": "genesis",
        "tip": pins.tip(),
        "hdr": genesis_hdr,
        "height": first.height if first is not None else -1,
        "count": 1 if first is not None else 0,
        "total": len(rows),
        "name": first.name if first else "",
        "rel": first.rel if first else "",
        "last_op": first.op if first else "",
        "sha256": first.sha256 if first else ZERO,
        "size": first.size if first else 0,
        "prev": prev,
        "at": first.at if first else "",
        "found": first is not None,
        "skills": [],
        "chain_ok": chain_ok,
        "empty": last is None,
        "genesis": genesis_hdr,
        "is_genesis": first is not None and prev == ZERO and first.height == 0,
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
        "exec": False,
    }
