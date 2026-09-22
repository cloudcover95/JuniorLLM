"""T29 — skill-pin children. Loopback only. Never fetch. Never exec."""
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


def _descendants(rows: list, root_hdr: str) -> list:
    kids: list = []
    seen: set[str] = set()
    frontier = [root_hdr]
    while frontier:
        cur = frontier.pop(0)
        nxt = [row for row in rows if row.prev == cur]
        nxt.sort(key=lambda row: row.height)
        for row in nxt:
            if row.hdr in seen:
                continue
            seen.add(row.hdr)
            kids.append(row)
            frontier.append(row.hdr)
    return kids


def skill_pin_children(hdr: str | None = None, root: str | None = None) -> dict:
    """T29 — list descendant SKILL_PINS.jsonl rows under HDR. Never body. Never fetch. Never exec."""
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
    parent = next((row for row in rows if row.hdr == want), None)
    kids: list = []
    is_tip = False
    if parent is None:
        issues.append("missing_hdr")
    else:
        kids = _descendants(rows, want)
        if not kids:
            is_tip = parent.hdr == pins.tip()
            if not is_tip:
                issues.append("missing_child")
                chain_ok = False

    seen_i: set[str] = set()
    uniq: list[str] = []
    for item in issues:
        if item not in seen_i:
            seen_i.add(item)
            uniq.append(item)

    first = kids[0] if kids else None
    last_kid = kids[-1] if kids else None
    found = bool(kids)
    ok = chain_ok and parent is not None and (found or is_tip)
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
        for row in kids
    ]
    return {
        "ok": ok,
        "issues": uniq,
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "op": "children",
        "tip": pins.tip(),
        "hdr": first.hdr if first else ZERO,
        "height": last_kid.height if last_kid is not None else -1,
        "count": len(kids),
        "total": len(rows),
        "name": first.name if first else "",
        "rel": first.rel if first else "",
        "last_op": last_kid.op if last_kid else "",
        "sha256": first.sha256 if first else ZERO,
        "size": first.size if first else 0,
        "prev": first.prev if first else ZERO,
        "at": first.at if first else "",
        "parent_hdr": want,
        "parent_height": parent.height if parent is not None else -1,
        "parent_op": parent.op if parent else "",
        "found": found,
        "entries": entries,
        "skills": [],
        "chain_ok": chain_ok,
        "empty": last is None,
        "genesis": rows[0].hdr if rows else ZERO,
        "is_tip": is_tip,
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
        "exec": False,
    }
