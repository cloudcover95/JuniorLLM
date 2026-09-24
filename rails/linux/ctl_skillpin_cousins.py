"""T32 — skill-pin cousins. Loopback only. Never fetch. Never exec."""
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


def _cousins(rows: list, child_hdr: str, zero: str) -> tuple[list, list[str]]:
    by_hdr = {row.hdr: row for row in rows}
    kid = by_hdr.get(child_hdr)
    if kid is None:
        return [], ["missing_hdr"]
    parent_hdr = kid.prev
    if parent_hdr == zero:
        return [], []
    parent = by_hdr.get(parent_hdr)
    if parent is None:
        return [], ["missing_parent"]
    uncles = [row for row in rows if row.prev == parent.prev and row.hdr != parent.hdr]
    uncle_hdrs = {row.hdr for row in uncles}
    peers = [row for row in rows if row.prev in uncle_hdrs and row.hdr != kid.hdr]
    peers.sort(key=lambda row: row.height)
    return peers, []


def skill_pin_cousins(hdr: str | None = None, root: str | None = None) -> dict:
    """T32 — list cousin SKILL_PINS.jsonl rows of HDR. Never body. Never fetch. Never exec."""
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
    peers: list = []
    is_only = False
    if child is None:
        issues.append("missing_hdr")
    else:
        peers, walk_issues = _cousins(rows, want, ZERO)
        issues.extend(walk_issues)
        if not peers:
            is_only = True

    seen_i: set[str] = set()
    uniq: list[str] = []
    for item in issues:
        if item not in seen_i:
            seen_i.add(item)
            uniq.append(item)

    first = peers[0] if peers else None
    last_peer = peers[-1] if peers else None
    found = bool(peers)
    ok = chain_ok and child is not None and (found or is_only)
    parent_hdr = child.prev if child is not None else ZERO
    parent = next((row for row in rows if row.hdr == parent_hdr), None) if child is not None else None
    grand_hdr = parent.prev if parent is not None else ZERO
    grand = next((row for row in rows if row.hdr == grand_hdr), None) if parent is not None else None
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
        for row in peers
    ]
    return {
        "ok": ok,
        "issues": uniq,
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "op": "cousins",
        "tip": pins.tip(),
        "hdr": first.hdr if first else ZERO,
        "height": last_peer.height if last_peer is not None else -1,
        "count": len(peers),
        "total": len(rows),
        "name": first.name if first else "",
        "rel": first.rel if first else "",
        "last_op": last_peer.op if last_peer else "",
        "sha256": first.sha256 if first else ZERO,
        "size": first.size if first else 0,
        "prev": first.prev if first else ZERO,
        "at": first.at if first else "",
        "child_hdr": want,
        "child_height": child.height if child is not None else -1,
        "child_op": child.op if child else "",
        "parent_hdr": parent_hdr if child is not None else ZERO,
        "parent_height": parent.height if parent is not None else -1,
        "parent_op": parent.op if parent else "",
        "grandparent_hdr": grand_hdr if parent is not None else ZERO,
        "grandparent_height": grand.height if grand is not None else -1,
        "grandparent_op": grand.op if grand else "",
        "found": found,
        "entries": entries,
        "skills": [],
        "chain_ok": chain_ok,
        "empty": last is None,
        "genesis": rows[0].hdr if rows else ZERO,
        "is_only": is_only,
        "is_genesis": bool(child is not None and child.prev == ZERO and child.height == 0),
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
        "exec": False,
    }
