"""juniorctl kinship wrappers T32–T42. Loopback only. Never fetch. Never exec."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path() -> None:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def skill_pin_cousins(hdr: str | None = None, root: str | None = None) -> dict:
    """T32 — pin-log cousin rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_cousins import skill_pin_cousins as impl

    return impl(hdr, root)


def skill_pin_uncles(hdr: str | None = None, root: str | None = None) -> dict:
    """T33 — pin-log uncle rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_uncles import skill_pin_uncles as impl

    return impl(hdr, root)


def skill_pin_nephews(hdr: str | None = None, root: str | None = None) -> dict:
    """T34 — pin-log nephew rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_nephews import skill_pin_nephews as impl

    return impl(hdr, root)


def skill_pin_grandchildren(hdr: str | None = None, root: str | None = None) -> dict:
    """T35 — pin-log grandchild rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_grandchildren import skill_pin_grandchildren as impl

    return impl(hdr, root)


def skill_pin_great_grandchildren(hdr: str | None = None, root: str | None = None) -> dict:
    """T36 — pin-log great-grandchild rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_great_grandchildren import skill_pin_great_grandchildren as impl

    return impl(hdr, root)


def skill_pin_great_great_grandchildren(hdr: str | None = None, root: str | None = None) -> dict:
    """T37 — pin-log great-great-grandchild rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_great_great_grandchildren import (
        skill_pin_great_great_grandchildren as impl,
    )

    return impl(hdr, root)


def skill_pin_2nd_cousins(hdr: str | None = None, root: str | None = None) -> dict:
    """T38 — pin-log 2nd-cousin rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_2nd_cousins import skill_pin_2nd_cousins as impl

    return impl(hdr, root)


def skill_pin_first_cousins_once_removed(hdr: str | None = None, root: str | None = None) -> dict:
    """T39 — pin-log 1C1R rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_first_cousins_once_removed import (
        skill_pin_first_cousins_once_removed as impl,
    )

    return impl(hdr, root)


def skill_pin_2nd_cousins_once_removed(hdr: str | None = None, root: str | None = None) -> dict:
    """T40 — pin-log 2C1R rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_2nd_cousins_once_removed import (
        skill_pin_2nd_cousins_once_removed as impl,
    )

    return impl(hdr, root)


def skill_pin_third_cousins(hdr: str | None = None, root: str | None = None) -> dict:
    """T41 — pin-log third-cousin rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_third_cousins import skill_pin_third_cousins as impl

    return impl(hdr, root)


def skill_pin_third_cousins_once_removed(hdr: str | None = None, root: str | None = None) -> dict:
    """T42 — pin-log 3C1R rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_third_cousins_once_removed import (
        skill_pin_third_cousins_once_removed as impl,
    )

    return impl(hdr, root)
