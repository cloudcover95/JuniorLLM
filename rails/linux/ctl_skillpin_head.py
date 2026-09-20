"""T24 — skill-pin head N. Loopback only. Never fetch. Never exec."""
from __future__ import annotations

from pathlib import Path

from rails.linux.ctl_skillpin import _denied, _guard_root

DEFAULT_N = 8


def _guard_n(text: str | None, raw: str) -> tuple[int | None, dict | None]:
    token = (text or "").strip()
    if not token:
        return DEFAULT_N, None
    if token.startswith("-") or not token.isdigit():
        return None, _denied("bad_n", raw)
    n = int(token)
    if n < 1:
        return None, _denied("bad_n", raw)
    return n, None


def skill_pin_head(n: str | None = None, root: str | None = None) -> dict:
    """T24 — list first N SKILL_PINS.jsonl rows under a root. Never body. Never fetch. Never exec."""
    from junior_aie.skill_pin import ZERO, SkillDenied, SkillPins

    raw, err = _guard_root(root)
    if err:
        return err
    want, bad = _guard_n(n, raw)
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
    window = rows[:want] if rows else []
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
        "ok": chain_ok,
        "issues": issues,
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "op": "head",
        "tip": pins.tip(),
        "hdr": tip_row.hdr if tip_row else ZERO,
        "height": tip_row.height if tip_row is not None else -1,
        "n": want,
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
        "genesis": rows[0].hdr if rows else ZERO,
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
        "exec": False,
    }
