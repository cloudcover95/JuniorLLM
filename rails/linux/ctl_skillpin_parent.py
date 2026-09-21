"""T27 — skill-pin parent. Loopback only. Never fetch. Never exec."""
from __future__ import annotations

from pathlib import Path

from rails.linux.ctl_skillpin import _denied, _guard_root


def _guard_hdr(text: str | None, raw: str) -> tuple[str | None, dict | None]:
    token = (text or "").strip().lower()
    if not token:
        return None, _denied("empty_hdr", raw)
    if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
        return None, _denied("bad_hdr", raw)
    return token, None


def skill_pin_parent(hdr: str | None = None, root: str | None = None) -> dict:
    """T27 — report parent SKILL_PINS.jsonl row of HDR. Never body. Never fetch. Never exec."""
    from junior_aie.skill_pin import ZERO, SkillDenied, SkillPins

    raw, err = _guard_root(root)
    if err:
        return err
    want, bad = _guard_hdr(hdr, raw)
    if bad:
        return bad
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
    child = next((row for row in rows if row.hdr == want), None)
    parent = None
    is_genesis = False
    if child is None:
        issues.append("missing_hdr")
    elif child.prev == ZERO:
        is_genesis = child.height == 0
        if not is_genesis:
            issues.append("genesis_prev")
            chain_ok = False
    else:
        parent = next((row for row in rows if row.hdr == child.prev), None)
        if parent is None:
            issues.append("missing_parent")
            chain_ok = False

    seen: set[str] = set()
    uniq: list[str] = []
    for item in issues:
        if item not in seen:
            seen.add(item)
            uniq.append(item)

    hit = parent
    found = hit is not None
    ok = chain_ok and child is not None and (found or is_genesis)
    return {
        "ok": ok,
        "issues": uniq,
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "op": "parent",
        "tip": pins.tip(),
        "hdr": hit.hdr if hit else ZERO,
        "height": hit.height if hit is not None else -1,
        "count": 1 if found else 0,
        "total": len(rows),
        "name": hit.name if hit else "",
        "rel": hit.rel if hit else "",
        "last_op": hit.op if hit else "",
        "sha256": hit.sha256 if hit else ZERO,
        "size": hit.size if hit else 0,
        "prev": hit.prev if hit else ZERO,
        "at": hit.at if hit else "",
        "child_hdr": want,
        "child_height": child.height if child is not None else -1,
        "child_op": child.op if child else "",
        "found": found,
        "skills": [],
        "chain_ok": chain_ok,
        "empty": last is None,
        "genesis": rows[0].hdr if rows else ZERO,
        "is_genesis": is_genesis,
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
        "exec": False,
    }
