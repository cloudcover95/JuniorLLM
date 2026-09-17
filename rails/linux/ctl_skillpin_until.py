"""T18 — skill-pin until HDR. Loopback only. Never fetch. Never exec."""
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


def skill_pin_until(hdr: str | None = None, root: str | None = None) -> dict:
    """T18 — list SKILL_PINS.jsonl rows from genesis through HDR. Never body. Never fetch. Never exec."""
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
    idx = next((i for i, row in enumerate(rows) if row.hdr == want), None)
    if idx is None:
        issues.append("missing_hdr")
        window = []
    else:
        window = rows[: idx + 1]
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
        "op": "until",
        "tip": pins.tip(),
        "hdr": tip_row.hdr if tip_row is not None and window else want,
        "height": tip_row.height if window and tip_row is not None else -1,
        "to_hdr": want,
        "to_height": tip_row.height if window and tip_row is not None else -1,
        "count": len(window),
        "total": len(rows),
        "name": first.name if first else "",
        "rel": first.rel if first else "",
        "last_op": tip_row.op if window and tip_row else "",
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
