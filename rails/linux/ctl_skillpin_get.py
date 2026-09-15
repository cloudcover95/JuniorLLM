"""T14 — skill-pin get HEIGHT. Loopback only. Never fetch. Never exec."""
from __future__ import annotations

from pathlib import Path

from rails.linux.ctl_skillpin import _denied, _guard_root


def _guard_height(text: str | None, raw: str) -> tuple[int | None, dict | None]:
    token = (text or "").strip()
    if not token:
        return None, _denied("empty_height", raw)
    if token.startswith("-") or not token.isdigit():
        return None, _denied("bad_height", raw)
    return int(token), None


def skill_pin_get(height: str | None = None, root: str | None = None) -> dict:
    """T14 — get one SKILL_PINS.jsonl row by HEIGHT. Never body. Never fetch. Never exec."""
    from junior_aie.skill_pin import ZERO, SkillDenied, SkillPins

    raw, err = _guard_root(root)
    if err:
        return err
    want, bad = _guard_height(height, raw)
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
    hit = next((row for row in rows if row.height == want), None)
    if hit is None:
        issues.append("missing_height")
    seen: set[str] = set()
    uniq: list[str] = []
    for item in issues:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return {
        "ok": chain_ok and hit is not None,
        "issues": uniq,
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "op": "get",
        "tip": pins.tip(),
        "hdr": hit.hdr if hit else ZERO,
        "height": hit.height if hit is not None else want,
        "count": len(rows),
        "name": hit.name if hit else "",
        "rel": hit.rel if hit else "",
        "last_op": hit.op if hit else "",
        "sha256": hit.sha256 if hit else ZERO,
        "size": hit.size if hit else 0,
        "prev": hit.prev if hit else ZERO,
        "at": hit.at if hit else "",
        "found": hit is not None,
        "skills": [],
        "chain_ok": chain_ok,
        "empty": last is None,
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
        "exec": False,
    }
