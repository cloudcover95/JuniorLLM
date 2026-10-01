"""T44 — skill-pin fourth-cousins-once-removed. Loopback only. Never fetch. Never exec."""
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


def _fourth_cousins_once_removed(rows: list, child_hdr: str, zero: str) -> tuple[list, list[str]]:
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
    grand_hdr = parent.prev
    down: list = []
    if grand_hdr != zero:
        grand = by_hdr.get(grand_hdr)
        if grand is None:
            return [], ["missing_grandparent"]
        great_hdr = grand.prev
        if great_hdr != zero:
            great = by_hdr.get(great_hdr)
            if great is None:
                return [], ["missing_great_grandparent"]
            ggreat_hdr = great.prev
            if ggreat_hdr != zero:
                ggreat = by_hdr.get(ggreat_hdr)
                if ggreat is None:
                    return [], ["missing_great_great_grandparent"]
                gggreat_hdr = ggreat.prev
                if gggreat_hdr != zero:
                    gggreat = by_hdr.get(gggreat_hdr)
                    if gggreat is None:
                        return [], ["missing_great_great_great_grandparent"]
                    gggreat_uncles = [
                        row for row in rows if row.prev == gggreat_hdr and row.hdr != ggreat.hdr
                    ]
                    uncle_hdrs = {row.hdr for row in gggreat_uncles}
                    ggreat_cousins = [row for row in rows if row.prev in uncle_hdrs]
                    ggreat_cousin_hdrs = {row.hdr for row in ggreat_cousins}
                    great_2nd = [row for row in rows if row.prev in ggreat_cousin_hdrs]
                    great_2nd_hdrs = {row.hdr for row in great_2nd}
                    parent_3rd = [row for row in rows if row.prev in great_2nd_hdrs]
                    parent_3rd_hdrs = {row.hdr for row in parent_3rd}
                    fourth = [
                        row for row in rows if row.prev in parent_3rd_hdrs and row.hdr != kid.hdr
                    ]
                    fourth_hdrs = {row.hdr for row in fourth}
                    down = [row for row in rows if row.prev in fourth_hdrs]
    parent_4th: list = []
    if grand_hdr != zero:
        grand = by_hdr.get(grand_hdr)
        if grand is None:
            return [], ["missing_grandparent"]
        great_hdr = grand.prev
        if great_hdr != zero:
            great = by_hdr.get(great_hdr)
            if great is None:
                return [], ["missing_great_grandparent"]
            ggreat_hdr = great.prev
            if ggreat_hdr != zero:
                ggreat = by_hdr.get(ggreat_hdr)
                if ggreat is None:
                    return [], ["missing_great_great_grandparent"]
                gggreat_hdr = ggreat.prev
                if gggreat_hdr != zero:
                    gggreat = by_hdr.get(gggreat_hdr)
                    if gggreat is None:
                        return [], ["missing_great_great_great_grandparent"]
                    gggggreat_hdr = gggreat.prev
                    if gggggreat_hdr != zero:
                        gggggreat = by_hdr.get(gggggreat_hdr)
                        if gggggreat is None:
                            return [], ["missing_great_great_great_great_grandparent"]
                        gggggreat_uncles = [
                            row for row in rows if row.prev == gggggreat_hdr and row.hdr != gggreat.hdr
                        ]
                        uncle_hdrs = {row.hdr for row in gggggreat_uncles}
                        gggreat_cousins = [row for row in rows if row.prev in uncle_hdrs]
                        gggreat_cousin_hdrs = {row.hdr for row in gggreat_cousins}
                        ggreat_2nd = [row for row in rows if row.prev in gggreat_cousin_hdrs]
                        ggreat_2nd_hdrs = {row.hdr for row in ggreat_2nd}
                        great_3rd = [row for row in rows if row.prev in ggreat_2nd_hdrs]
                        great_3rd_hdrs = {row.hdr for row in great_3rd}
                        parent_4th = [row for row in rows if row.prev in great_3rd_hdrs]
    seen: set[str] = set()
    peers: list = []
    for row in down + parent_4th:
        if row.hdr == kid.hdr or row.hdr in seen:
            continue
        seen.add(row.hdr)
        peers.append(row)
    peers.sort(key=lambda row: row.height)
    return peers, []


def skill_pin_fourth_cousins_once_removed(hdr: str | None = None, root: str | None = None) -> dict:
    """T44 — list 4C1R SKILL_PINS.jsonl rows of HDR. Never body. Never fetch. Never exec."""
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
        peers, walk_issues = _fourth_cousins_once_removed(rows, want, ZERO)
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
    gggggreat = next((row for row in rows if row.hdr == gggggreat_hdr), None) if gggreat is not None else None
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
        "op": "fourth-cousins-once-removed",
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
        "great_grandparent_height": great.height if great is not None else -1,
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
