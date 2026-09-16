"""T16 — skill-pin range FROM TO. Loopback only. Never fetch. Never exec."""
from __future__ import annotations

from pathlib import Path

from rails.linux.ctl_skillpin import _denied, _guard_root


def _guard_bound(text: str | None, raw: str, empty: str, bad: str) -> tuple[int | None, dict | None]:
    token = (text or "").strip()
    if not token:
        return None, _denied(empty, raw)
    if token.startswith("-") or not token.isdigit():
        return None, _denied(bad, raw)
    return int(token), None


def skill_pin_range(
    start: str | None = None,
    stop: str | None = None,
    root: str | None = None,
) -> dict:
    """T16 — list SKILL_PINS.jsonl rows from HEIGHT FROM through TO. Never body. Never fetch. Never exec."""
    from junior_aie.skill_pin import ZERO, SkillDenied, SkillPins

    raw, err = _guard_root(root)
    if err:
        return err
    lo, bad_lo = _guard_bound(start, raw, "empty_from", "bad_from")
    if bad_lo:
        return bad_lo
    hi, bad_hi = _guard_bound(stop, raw, "empty_to", "bad_to")
    if bad_hi:
        return bad_hi
    if lo > hi:
        return _denied("inverted_range", raw)
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
    window = [row for row in rows if lo <= row.height <= hi]
    if not window:
        issues.append("missing_range")
    seen: set[str] = set()
    uniq: list[str] = []
    for item in issues:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    first = window[0] if window else None
    tip_row = window[-1] if window else last
    entries = [
        {
            "height": row.height,
            "prev": row.prev,
            "op": row.op,
            "name": row.name,
            "rel": row.rel,
            "sha256": row.sha256,
            "size": row.size,
            "at": row.at,
            "hdr": row.hdr,
        }
        for row in window
    ]
    return {
        "ok": chain_ok and bool(window),
        "issues": uniq,
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "op": "range",
        "tip": pins.tip(),
        "hdr": tip_row.hdr if tip_row else ZERO,
        "height": tip_row.height if tip_row is not None else -1,
        "from": lo,
        "to": hi,
        "count": len(window),
        "total": len(rows),
        "name": first.name if first else "",
        "rel": first.rel if first else "",
        "last_op": tip_row.op if tip_row else "",
        "sha256": first.sha256 if first else ZERO,
        "size": first.size if first else 0,
        "prev": first.prev if first else ZERO,
        "at": first.at if first else "",
        "found": bool(window),
        "entries": entries,
        "skills": [],
        "chain_ok": chain_ok,
        "empty": last is None,
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
        "exec": False,
    }
