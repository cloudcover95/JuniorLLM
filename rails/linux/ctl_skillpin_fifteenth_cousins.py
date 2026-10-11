"""T64 — skill-pin fifteenth-cousins. Loopback only. Never fetch. Never exec."""
from __future__ import annotations

from pathlib import Path

from rails.linux.ctl_skillpin import _denied, _guard_root

_MISSING = (
    "missing_parent",
    "missing_grandparent",
    "missing_great_grandparent",
    "missing_great_great_grandparent",
    "missing_great_great_great_grandparent",
    "missing_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_great_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_great_great_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_great_great_great_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_great_great_great_great_great_great_great_grandparent",
    "missing_great_great_great_great_great_great_great_great_great_great_great_great_great_grandparent",
)


def _guard_hdr(text: str | None, raw: str) -> tuple[str | None, dict | None]:
    token = (text or "").strip().lower()
    if not token:
        return None, _denied("empty_hdr", raw)
    if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
        return None, _denied("bad_hdr", raw)
    return token, None


def _fifteenth_cousins(rows: list, child_hdr: str, zero: str) -> tuple[list, list[str]]:
    """Fifteenth cousins share a 14-great-grandparent: 16 up, then 15 down the other lines."""
    by_hdr = {row.hdr: row for row in rows}
    kid = by_hdr.get(child_hdr)
    if kid is None:
        return [], ["missing_hdr"]
    hdr = child_hdr
    chain: list = []
    for step in range(16):
        row = by_hdr.get(hdr)
        if row is None:
            label = _MISSING[step] if step < len(_MISSING) else "missing_ancestor"
            return [], [label]
        if row.prev == zero:
            return [], []
        nxt = by_hdr.get(row.prev)
        if nxt is None:
            label = _MISSING[step] if step < len(_MISSING) else "missing_ancestor"
            return [], [label]
        chain.append(nxt)
        hdr = nxt.hdr
    shared = chain[-1]
    line_child = chain[-2]
    uncles = [row for row in rows if row.prev == shared.hdr and row.hdr != line_child.hdr]
    current = {row.hdr for row in uncles}
    peers: list = []
    for _ in range(15):
        peers = [row for row in rows if row.prev in current]
        current = {row.hdr for row in peers}
    peers = [row for row in peers if row.hdr != kid.hdr]
    peers.sort(key=lambda row: row.height)
    return peers, []


def skill_pin_fifteenth_cousins(hdr: str | None = None, root: str | None = None) -> dict:
    """T64 — list fifteenth-cousin SKILL_PINS.jsonl rows of HDR. Never body. Never fetch. Never exec."""
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
        peers, walk_issues = _fifteenth_cousins(rows, want, ZERO)
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
    # Walk up to 15 ancestors for the return fields (15th cousins share 14-great).
    ancestors = []
    cur = child
    for _ in range(16):
        if cur is None:
            ancestors.append(None)
            continue
        prev_hdr = cur.prev
        nxt = next((row for row in rows if row.hdr == prev_hdr), None) if cur is not None else None
        ancestors.append(nxt)
        cur = nxt
    # Map to named fields: parent is ancestors[0], grand parents[1], ..., g14 is ancestors[15]
    parent = ancestors[0] if len(ancestors) > 0 else None
    grand = ancestors[1] if len(ancestors) > 1 else None
    great = ancestors[2] if len(ancestors) > 2 else None
    ggreat = ancestors[3] if len(ancestors) > 3 else None
    gggreat = ancestors[4] if len(ancestors) > 4 else None
    gggggreat = ancestors[5] if len(ancestors) > 5 else None
    ggggggreat = ancestors[6] if len(ancestors) > 6 else None
    gggggggreat = ancestors[7] if len(ancestors) > 7 else None
    ggggggggreat = ancestors[8] if len(ancestors) > 8 else None
    gggggggggreat = ancestors[9] if len(ancestors) > 9 else None
    ggggggggggreat = ancestors[10] if len(ancestors) > 10 else None
    g10 = ancestors[11] if len(ancestors) > 11 else None
    g11 = ancestors[12] if len(ancestors) > 12 else None
    g12 = ancestors[13] if len(ancestors) > 13 else None
    g13 = ancestors[14] if len(ancestors) > 14 else None
    g14 = ancestors[15] if len(ancestors) > 15 else None
    parent_hdr = parent.hdr if parent is not None else ZERO
    grand_hdr = grand.hdr if grand is not None else ZERO
    great_hdr = great.hdr if great is not None else ZERO
    ggreat_hdr = ggreat.hdr if ggreat is not None else ZERO
    gggreat_hdr = gggreat.hdr if gggreat is not None else ZERO
    gggggreat_hdr = gggggreat.hdr if gggggreat is not None else ZERO
    ggggggreat_hdr = ggggggreat.hdr if ggggggreat is not None else ZERO
    gggggggreat_hdr = gggggggreat.hdr if gggggggreat is not None else ZERO
    ggggggggreat_hdr = ggggggggreat.hdr if ggggggggreat is not None else ZERO
    gggggggggreat_hdr = gggggggggreat.hdr if gggggggggreat is not None else ZERO
    ggggggggggreat_hdr = ggggggggggreat.hdr if ggggggggggreat is not None else ZERO
    g10_hdr = g10.hdr if g10 is not None else ZERO
    g11_hdr = g11.hdr if g11 is not None else ZERO
    g12_hdr = g12.hdr if g12 is not None else ZERO
    g13_hdr = g13.hdr if g13 is not None else ZERO
    g14_hdr = g14.hdr if g14 is not None else ZERO
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
        "op": "fifteenth-cousins",
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
        "great_grandparent_hdr": great_hdr if grand is not None else ZERO,
        "great_grandparent_height": great.height if grand is not None else -1,
        "great_grandparent_op": great.op if great else "",
        "great_great_grandparent_hdr": ggreat_hdr if great is not None else ZERO,
        "great_great_grandparent_height": ggreat.height if ggreat is not None else -1,
        "great_great_grandparent_op": ggreat.op if ggreat else "",
        "great_great_great_grandparent_hdr": gggreat_hdr if ggreat is not None else ZERO,
        "great_great_great_grandparent_height": gggreat.height if gggreat is not None else -1,
        "great_great_great_grandparent_op": gggreat.op if gggreat else "",
        "great_great_great_great_grandparent_hdr": gggggreat_hdr if gggreat is not None else ZERO,
        "great_great_great_great_grandparent_height": gggggreat.height if gggggreat is not None else -1,
        "great_great_great_great_grandparent_op": gggggreat.op if gggggreat else "",
        "great_great_great_great_great_grandparent_hdr": ggggggreat_hdr if ggggggreat is not None else ZERO,
        "great_great_great_great_great_grandparent_height": (
            ggggggreat.height if ggggggreat is not None else -1
        ),
        "great_great_great_great_great_grandparent_op": ggggggreat.op if ggggggreat else "",
        "great_great_great_great_great_great_grandparent_hdr": (
            gggggggreat_hdr if ggggggreat is not None else ZERO
        ),
        "great_great_great_great_great_great_grandparent_height": (
            gggggggreat.height if gggggggreat is not None else -1
        ),
        "great_great_great_great_great_great_grandparent_op": gggggggreat.op if gggggggreat else "",
        "great_great_great_great_great_great_great_grandparent_hdr": (
            ggggggggreat_hdr if gggggggreat is not None else ZERO
        ),
        "great_great_great_great_great_great_great_grandparent_height": (
            ggggggggreat.height if ggggggggreat is not None else -1
        ),
        "great_great_great_great_great_great_great_grandparent_op": (
            ggggggggreat.op if ggggggggreat else ""
        ),
        "great_great_great_great_great_great_great_great_grandparent_hdr": (
            gggggggggreat_hdr if ggggggggreat is not None else ZERO
        ),
        "great_great_great_great_great_great_great_great_grandparent_height": (
            gggggggggreat.height if gggggggggreat is not None else -1
        ),
        "great_great_great_great_great_great_great_great_grandparent_op": (
            gggggggggreat.op if gggggggggreat else ""
        ),
        "great_great_great_great_great_great_great_great_great_grandparent_hdr": (
            ggggggggggreat_hdr if gggggggggreat is not None else ZERO
        ),
        "great_great_great_great_great_great_great_great_great_grandparent_height": (
            ggggggggggreat.height if ggggggggggreat is not None else -1
        ),
        "great_great_great_great_great_great_great_great_great_grandparent_op": (
            ggggggggggreat.op if ggggggggggreat else ""
        ),
        "great_great_great_great_great_great_great_great_great_great_grandparent_hdr": (
            g10.hdr if g10 is not None else ZERO
        ),
        "great_great_great_great_great_great_great_great_great_great_grandparent_height": (
            g10.height if g10 is not None else -1
        ),
        "great_great_great_great_great_great_great_great_great_great_grandparent_op": (
            g10.op if g10 else ""
        ),
        "great_great_great_great_great_great_great_great_great_great_great_grandparent_hdr": (
            g11.hdr if g11 is not None else ZERO
        ),
        "great_great_great_great_great_great_great_great_great_great_great_grandparent_height": (
            g11.height if g11 is not None else -1
        ),
        "great_great_great_great_great_great_great_great_great_great_great_grandparent_op": (
            g11.op if g11 else ""
        ),
        "great_great_great_great_great_great_great_great_great_great_great_great_grandparent_hdr": (
            g12.hdr if g12 is not None else ZERO
        ),
        "great_great_great_great_great_great_great_great_great_great_great_great_grandparent_height": (
            g12.height if g12 is not None else -1
        ),
        "great_great_great_great_great_great_great_great_great_great_great_great_grandparent_op": (
            g12.op if g12 else ""
        ),
        "great_great_great_great_great_great_great_great_great_great_great_great_great_grandparent_hdr": (
            g13.hdr if g13 is not None else ZERO
        ),
        "great_great_great_great_great_great_great_great_great_great_great_great_great_grandparent_height": (
            g13.height if g13 is not None else -1
        ),
        "great_great_great_great_great_great_great_great_great_great_great_great_great_grandparent_op": (
            g13.op if g13 else ""
        ),
        "great_great_great_great_great_great_great_great_great_great_great_great_great_grandparent_hdr": (
            g14.hdr if g14 is not None else ZERO
        ),
        "great_great_great_great_great_great_great_great_great_great_great_great_great_grandparent_height": (
            g14.height if g14 is not None else -1
        ),
        "great_great_great_great_great_great_great_great_great_great_great_great_great_grandparent_op": (
            g14.op if g14 else ""
        ),
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
