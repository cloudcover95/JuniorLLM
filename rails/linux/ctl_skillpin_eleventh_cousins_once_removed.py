"""T58 — skill-pin eleventh-cousins-once-removed. Loopback only. Never fetch. Never exec."""
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
)


def _guard_hdr(text: str | None, raw: str) -> tuple[str | None, dict | None]:
    token = (text or "").strip().lower()
    if not token:
        return None, _denied("empty_hdr", raw)
    if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
        return None, _denied("bad_hdr", raw)
    return token, None


def _walk(by_hdr: dict, start_hdr: str, zero: str, depth: int) -> tuple[list, str | None]:
    chain: list = []
    hdr = start_hdr
    for step in range(depth):
        if hdr == zero:
            return chain, None
        row = by_hdr.get(hdr)
        if row is None:
            label = _MISSING[step] if step < len(_MISSING) else "missing_ancestor"
            return chain, label
        chain.append(row)
        hdr = row.prev
    return chain, None


def _descend(rows: list, uncle_hdrs: set[str], generations: int) -> list:
    current = uncle_hdrs
    peers: list = []
    for _ in range(generations):
        peers = [row for row in rows if row.prev in current]
        current = {row.hdr for row in peers}
    return peers


def _eleventh_cousins_once_removed(rows: list, child_hdr: str, zero: str) -> tuple[list, list[str]]:
    """11C1R: children of eleventh cousins, plus eleventh cousins of the parent."""
    by_hdr = {row.hdr: row for row in rows}
    kid = by_hdr.get(child_hdr)
    if kid is None:
        return [], ["missing_hdr"]
    if kid.prev == zero:
        return [], []

    # Children of eleventh cousins: shared 10-great-grandparent (12 up from parent), then 11 down, then one more.
    down: list = []
    down_chain, down_issue = _walk(by_hdr, kid.prev, zero, 12)
    if down_issue:
        return [], [down_issue]
    if len(down_chain) == 12:
        shared = down_chain[-1]
        line_child = down_chain[-2]
        uncles = [row for row in rows if row.prev == shared.hdr and row.hdr != line_child.hdr]
        eleventh = _descend(rows, {row.hdr for row in uncles}, 11)
        eleventh_hdrs = {row.hdr for row in eleventh if row.hdr != kid.hdr}
        down = [row for row in rows if row.prev in eleventh_hdrs]

    # Eleventh cousins of the parent: shared 11-great-grandparent (13 up from parent), then 11 down.
    parent_11th: list = []
    up_chain, up_issue = _walk(by_hdr, kid.prev, zero, 13)
    if up_issue:
        return [], [up_issue]
    if len(up_chain) == 13:
        shared = up_chain[-1]
        line_child = up_chain[-2]
        uncles = [row for row in rows if row.prev == shared.hdr and row.hdr != line_child.hdr]
        parent_11th = _descend(rows, {row.hdr for row in uncles}, 11)

    seen: set[str] = set()
    peers: list = []
    for row in down + parent_11th:
        if row.hdr == kid.hdr or row.hdr in seen:
            continue
        seen.add(row.hdr)
        peers.append(row)
    peers.sort(key=lambda row: row.height)
    return peers, []


def skill_pin_eleventh_cousins_once_removed(hdr: str | None = None, root: str | None = None) -> dict:
    """T58 — list 11C1R SKILL_PINS.jsonl rows of HDR. Never body. Never fetch. Never exec."""
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
        peers, walk_issues = _eleventh_cousins_once_removed(rows, want, ZERO)
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
    great_hdr = grand.prev if grand is not None else ZERO
    great = next((row for row in rows if row.hdr == great_hdr), None) if grand is not None else None
    ggreat_hdr = great.prev if great is not None else ZERO
    ggreat = next((row for row in rows if row.hdr == ggreat_hdr), None) if great is not None else None
    gggreat_hdr = ggreat.prev if ggreat is not None else ZERO
    gggreat = next((row for row in rows if row.hdr == gggreat_hdr), None) if ggreat is not None else None
    gggggreat_hdr = gggreat.prev if gggreat is not None else ZERO
    gggggreat = (
        next((row for row in rows if row.hdr == gggggreat_hdr), None) if gggreat is not None else None
    )
    ggggggreat_hdr = gggggreat.prev if gggggreat is not None else ZERO
    ggggggreat = (
        next((row for row in rows if row.hdr == ggggggreat_hdr), None)
        if gggggreat is not None
        else None
    )
    gggggggreat_hdr = ggggggreat.prev if ggggggreat is not None else ZERO
    gggggggreat = (
        next((row for row in rows if row.hdr == gggggggreat_hdr), None)
        if ggggggreat is not None
        else None
    )
    ggggggggreat_hdr = gggggggreat.prev if gggggggreat is not None else ZERO
    ggggggggreat = (
        next((row for row in rows if row.hdr == ggggggggreat_hdr), None)
        if gggggggreat is not None
        else None
    )
    gggggggggreat_hdr = ggggggggreat.prev if ggggggggreat is not None else ZERO
    gggggggggreat = (
        next((row for row in rows if row.hdr == gggggggggreat_hdr), None)
        if ggggggggreat is not None
        else None
    )
    ggggggggggreat_hdr = gggggggggreat.prev if gggggggggreat is not None else ZERO
    ggggggggggreat = (
        next((row for row in rows if row.hdr == ggggggggggreat_hdr), None)
        if gggggggggreat is not None
        else None
    )
    g10_hdr = ggggggggggreat.prev if ggggggggggreat is not None else ZERO
    g10 = (
        next((row for row in rows if row.hdr == g10_hdr), None)
        if ggggggggggreat is not None
        else None
    )
    g11_hdr = g10.prev if g10 is not None else ZERO
    g11 = next((row for row in rows if row.hdr == g11_hdr), None) if g10 is not None else None
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
        "op": "eleventh-cousins-once-removed",
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
