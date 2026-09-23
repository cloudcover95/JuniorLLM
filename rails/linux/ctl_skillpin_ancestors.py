"""T30 — skill-pin ancestors. Loopback only. Never fetch. Never exec."""
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


def _ancestors(rows: list, child_hdr: str, zero: str) -> tuple[list, list[str]]:
    by_hdr = {row.hdr: row for row in rows}
    kid = by_hdr.get(child_hdr)
    if kid is None:
        return [], ["missing_hdr"]
    chain: list = []
    seen: set[str] = set()
    cur = kid.prev
    issues: list[str] = []
    while cur and cur != zero:
        if cur in seen:
            issues.append("cycle")
            break
        seen.add(cur)
        row = by_hdr.get(cur)
        if row is None:
            issues.append("missing_parent")
            break
        chain.append(row)
        cur = row.prev
    return chain, issues


def skill_pin_ancestors(hdr: str | None = None, root: str | None = None) -> dict:
    """T30 — list ancestor SKILL_PINS.jsonl rows of HDR. Never body. Never fetch. Never exec."""
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
    elders: list = []
    is_genesis = False
    if child is None:
        issues.append("missing_hdr")
    else:
        elders, walk_issues = _ancestors(rows, want, ZERO)
        issues.extend(walk_issues)
        if walk_issues:
            chain_ok = False
        if not elders:
            is_genesis = child.prev == ZERO and child.height == 0
            if child.prev == ZERO and not is_genesis:
                issues.append("genesis_prev")
                chain_ok = False
            elif child.prev != ZERO and "missing_parent" not in issues:
                issues.append("missing_parent")
                chain_ok = False

    seen_i: set[str] = set()
    uniq: list[str] = []
    for item in issues:
        if item not in seen_i:
            seen_i.add(item)
            uniq.append(item)

    first = elders[0] if elders else None
    last_eld = elders[-1] if elders else None
    found = bool(elders)
    ok = chain_ok and child is not None and (found or is_genesis)
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
        for row in elders
    ]
    return {
        "ok": ok,
        "issues": uniq,
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "op": "ancestors",
        "tip": pins.tip(),
        "hdr": first.hdr if first else ZERO,
        "height": last_eld.height if last_eld is not None else -1,
        "count": len(elders),
        "total": len(rows),
        "name": first.name if first else "",
        "rel": first.rel if first else "",
        "last_op": last_eld.op if last_eld else "",
        "sha256": first.sha256 if first else ZERO,
        "size": first.size if first else 0,
        "prev": first.prev if first else ZERO,
        "at": first.at if first else "",
        "child_hdr": want,
        "child_height": child.height if child is not None else -1,
        "child_op": child.op if child else "",
        "found": found,
        "entries": entries,
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
