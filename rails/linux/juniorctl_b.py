from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _path() -> None:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


def skill_pin_tip(root: str | None = None) -> dict:
    """T11 — pin-log chain tip under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin import skill_pin_tip as impl

    return impl(root)


def skill_pin_log(root: str | None = None) -> dict:
    """T12 — pin-log chain rows under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin import skill_pin_log as impl

    return impl(root)


def skill_pin_height(root: str | None = None) -> dict:
    """T13 — pin-log chain height under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin import skill_pin_height as impl

    return impl(root)


def skill_pin_get(height: str | None = None, root: str | None = None) -> dict:
    """T14 — pin-log row by HEIGHT under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_get import skill_pin_get as impl

    return impl(height, root)


def skill_pin_at(hdr: str | None = None, root: str | None = None) -> dict:
    """T15 — pin-log row by HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_at import skill_pin_at as impl

    return impl(hdr, root)


def skill_pin_range(
    start: str | None = None,
    stop: str | None = None,
    root: str | None = None,
) -> dict:
    """T16 — pin-log rows FROM TO under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_range import skill_pin_range as impl

    return impl(start, stop, root)


def skill_pin_since(hdr: str | None = None, root: str | None = None) -> dict:
    """T17 — pin-log rows since HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_since import skill_pin_since as impl

    return impl(hdr, root)


def skill_pin_until(hdr: str | None = None, root: str | None = None) -> dict:
    """T18 — pin-log rows until HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_until import skill_pin_until as impl

    return impl(hdr, root)


def skill_pin_before(hdr: str | None = None, root: str | None = None) -> dict:
    """T19 — pin-log rows before HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_before import skill_pin_before as impl

    return impl(hdr, root)


def skill_pin_after(hdr: str | None = None, root: str | None = None) -> dict:
    """T20 — pin-log rows after HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_after import skill_pin_after as impl

    return impl(hdr, root)


def skill_pin_first(root: str | None = None) -> dict:
    """T21 — pin-log first row under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_first import skill_pin_first as impl

    return impl(root)


def skill_pin_last(root: str | None = None) -> dict:
    """T22 — pin-log last row under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_last import skill_pin_last as impl

    return impl(root)


def skill_pin_tail(n: str | None = None, root: str | None = None) -> dict:
    """T23 — pin-log last N rows under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_tail import skill_pin_tail as impl

    return impl(n, root)


def skill_pin_head(n: str | None = None, root: str | None = None) -> dict:
    """T24 — pin-log first N rows under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_head import skill_pin_head as impl

    return impl(n, root)


def skill_pin_count(root: str | None = None) -> dict:
    """T25 — pin-log row count under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_count import skill_pin_count as impl

    return impl(root)


def skill_pin_genesis(root: str | None = None) -> dict:
    """T26 — pin-log genesis row under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_genesis import skill_pin_genesis as impl

    return impl(root)


def skill_pin_parent(hdr: str | None = None, root: str | None = None) -> dict:
    """T27 — pin-log parent row of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_parent import skill_pin_parent as impl

    return impl(hdr, root)


def skill_pin_child(hdr: str | None = None, root: str | None = None) -> dict:
    """T28 — pin-log child row of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_child import skill_pin_child as impl

    return impl(hdr, root)


def skill_pin_children(hdr: str | None = None, root: str | None = None) -> dict:
    """T29 — pin-log descendant rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_children import skill_pin_children as impl

    return impl(hdr, root)


def skill_pin_ancestors(hdr: str | None = None, root: str | None = None) -> dict:
    """T30 — pin-log ancestor rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_ancestors import skill_pin_ancestors as impl

    return impl(hdr, root)


def skill_pin_siblings(hdr: str | None = None, root: str | None = None) -> dict:
    """T31 — pin-log sibling rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_siblings import skill_pin_siblings as impl

    return impl(hdr, root)
