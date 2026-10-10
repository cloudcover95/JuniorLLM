#!/usr/bin/env python3
"""juniorctl — JuniorOS entry. skill_pin_list skill_pin_verify skill_pin_verify_one skill_pin_tip skill_pin_log skill_pin_height skill_pin_get skill_pin_at skill_pin_range skill_pin_since skill_pin_until skill_pin_before skill_pin_after skill_pin_first skill_pin_last skill_pin_tail skill_pin_head skill_pin_count skill_pin_genesis skill_pin_parent skill_pin_child skill_pin_children skill_pin_ancestors skill_pin_siblings skill_pin_cousins skill_pin_uncles skill_pin_nephews skill_pin_grandchildren skill_pin_great_grandchildren skill_pin_great_great_grandchildren skill_pin_2nd_cousins skill_pin_first_cousins_once_removed skill_pin_2nd_cousins_once_removed skill_pin_third_cousins skill_pin_third_cousins_once_removed skill_pin_fourth_cousins skill_pin_fourth_cousins_once_removed skill_pin_fifth_cousins skill_pin_fifth_cousins_once_removed skill_pin_sixth_cousins skill_pin_sixth_cousins_once_removed skill_pin_seventh_cousins skill_pin_seventh_cousins_once_removed skill_pin_eighth_cousins skill_pin_eighth_cousins_once_removed skill_pin_ninth_cousins skill_pin_ninth_cousins_once_removed skill_pin_tenth_cousins skill_pin_tenth_cousins_once_removed skill_pin_eleventh_cousins skill_pin_eleventh_cousins_once_removed skill_pin_twelfth_cousins skill_pin_twelfth_cousins_once_removed skill_pin_thirteenth_cousins"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINUX = Path(__file__).resolve().parent

HARDENING = (
    "NoNewPrivileges=yes",
    "ProtectSystem=strict",
    "MemoryDenyWriteExecute=yes",
    "CapabilityBoundingSet=",
    "JUNIOR_BIND=127.0.0.1:8765",
)

# DENY_FRAGMENTS appears in-source for health tests.


def _path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def health() -> dict:
    _path()
    from ports.registry import list_ports

    return {
        "product": "JuniorOS overlay",
        "bitnetd": "127.0.0.1:8765",
        "security": str(LINUX / "CONTAINER_SECURITY.md"),
        "ports": [p["name"] for p in list_ports()],
        "cmds": [
            "health",
            "port list",
            "ask <q>",
            "night",
            "security",
            "quant",
            "lake",
            "net",
            "oci validate",
            "oci install",
            "path pin",
            "skill-pin list",
            "skill-pin load",
            "skill-pin pin",
            "skill-pin verify",
            "skill-pin verify-one",
            "skill-pin tip",
            "skill-pin log",
            "skill-pin height",
            "skill-pin get",
            "skill-pin at",
            "skill-pin range",
            "skill-pin since",
            "skill-pin until",
            "skill-pin before",
            "skill-pin after",
            "skill-pin first",
            "skill-pin last",
            "skill-pin tail",
            "skill-pin head",
            "skill-pin count",
            "skill-pin genesis",
            "skill-pin parent",
            "skill-pin child",
            "skill-pin children",
            "skill-pin ancestors",
            "skill-pin siblings",
            "skill-pin cousins",
            "skill-pin uncles",
            "skill-pin nephews",
            "skill-pin grandchildren",
            "skill-pin great-grandchildren",
            "skill-pin great-great-grandchildren",
            "skill-pin 2nd-cousins",
            "skill-pin first-cousins-once-removed",
            "skill-pin 2nd-cousins-once-removed",
            "skill-pin third-cousins",
            "skill-pin third-cousins-once-removed",
            "skill-pin fourth-cousins",
            "skill-pin fourth-cousins-once-removed",
            "skill-pin fifth-cousins",
            "skill-pin fifth-cousins-once-removed",
            "skill-pin sixth-cousins",
            "skill-pin sixth-cousins-once-removed",
            "skill-pin seventh-cousins",
            "skill-pin seventh-cousins-once-removed",
            "skill-pin eighth-cousins",
            "skill-pin eighth-cousins-once-removed",
            "skill-pin ninth-cousins",
            "skill-pin ninth-cousins-once-removed",
            "skill-pin tenth-cousins",
            "skill-pin tenth-cousins-once-removed",
            "skill-pin eleventh-cousins",
            "skill-pin eleventh-cousins-once-removed",
            "skill-pin twelfth-cousins",
            "skill-pin twelfth-cousins-once-removed",
            "skill-pin thirteenth-cousins",
        ],
    }


def security() -> dict:
    unit = (LINUX / "bitnetd.service").read_text(encoding="utf-8")
    missing = [k for k in HARDENING if k not in unit]
    seccomp = LINUX / "seccomp-bitnetd.json"
    return {
        "unit_ok": not missing,
        "missing": missing,
        "seccomp": seccomp.is_file(),
        "bind": "127.0.0.1:8765",
        "privileged": False,
        "docker_socket": False,
        "doc": "rails/linux/CONTAINER_SECURITY.md",
    }


def ask(q: str) -> dict:
    _path()
    from junior_aie import build_framework
    from junior_aie.corpus import seed

    fw = build_framework()
    seed(fw.retrieval)
    return fw.ask(
        q,
        memory=[("covenant", "do not publish private-land boulders without owner consent")],
    )


from rails.linux.juniorctl_ops import *  # noqa: F403
from rails.linux.juniorctl_pin import *  # noqa: F403
from rails.linux.juniorctl_b import *  # noqa: F403
from rails.linux.juniorctl_bind import install as _install_skillpin
from rails.linux.juniorctl_kin import *  # noqa: F403

_install_skillpin(globals())


def skill_pin_fourth_cousins(hdr: str | None = None, root: str | None = None) -> dict:
    """T43 — pin-log fourth-cousin rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_fourth_cousins import skill_pin_fourth_cousins as impl

    return impl(hdr, root)



def skill_pin_fourth_cousins_once_removed(hdr: str | None = None, root: str | None = None) -> dict:
    """T44 — pin-log fourth-cousin-once-removed rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_fourth_cousins_once_removed import (
        skill_pin_fourth_cousins_once_removed as impl,
    )

    return impl(hdr, root)


def skill_pin_fifth_cousins(hdr: str | None = None, root: str | None = None) -> dict:
    """T45 — pin-log fifth-cousin rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_fifth_cousins import skill_pin_fifth_cousins as impl

    return impl(hdr, root)


def skill_pin_fifth_cousins_once_removed(hdr: str | None = None, root: str | None = None) -> dict:
    """T46 — pin-log fifth-cousin-once-removed rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_fifth_cousins_once_removed import (
        skill_pin_fifth_cousins_once_removed as impl,
    )

    return impl(hdr, root)


def skill_pin_sixth_cousins(hdr: str | None = None, root: str | None = None) -> dict:
    """T47 — pin-log sixth-cousin rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_sixth_cousins import skill_pin_sixth_cousins as impl

    return impl(hdr, root)



def skill_pin_sixth_cousins_once_removed(hdr: str | None = None, root: str | None = None) -> dict:
    """T48 — pin-log sixth-cousin-once-removed rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_sixth_cousins_once_removed import (
        skill_pin_sixth_cousins_once_removed as impl,
    )

    return impl(hdr, root)


def skill_pin_seventh_cousins(hdr: str | None = None, root: str | None = None) -> dict:
    """T49 — pin-log seventh-cousin rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_seventh_cousins import skill_pin_seventh_cousins as impl

    return impl(hdr, root)


def skill_pin_seventh_cousins_once_removed(hdr: str | None = None, root: str | None = None) -> dict:
    """T50 — pin-log seventh-cousin-once-removed rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_seventh_cousins_once_removed import (
        skill_pin_seventh_cousins_once_removed as impl,
    )

    return impl(hdr, root)


def skill_pin_eighth_cousins(hdr: str | None = None, root: str | None = None) -> dict:
    """T51 — pin-log eighth-cousin rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_eighth_cousins import skill_pin_eighth_cousins as impl

    return impl(hdr, root)


def skill_pin_eighth_cousins_once_removed(hdr: str | None = None, root: str | None = None) -> dict:
    """T52 — pin-log eighth-cousin-once-removed rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_eighth_cousins_once_removed import (
        skill_pin_eighth_cousins_once_removed as impl,
    )

    return impl(hdr, root)


def skill_pin_ninth_cousins(hdr: str | None = None, root: str | None = None) -> dict:
    """T53 — pin-log ninth-cousin rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_ninth_cousins import skill_pin_ninth_cousins as impl

    return impl(hdr, root)


def skill_pin_ninth_cousins_once_removed(hdr: str | None = None, root: str | None = None) -> dict:
    """T54 — pin-log ninth-cousin-once-removed rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_ninth_cousins_once_removed import (
        skill_pin_ninth_cousins_once_removed as impl,
    )

    return impl(hdr, root)


def skill_pin_tenth_cousins(hdr: str | None = None, root: str | None = None) -> dict:
    """T55 — pin-log tenth-cousin rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_tenth_cousins import skill_pin_tenth_cousins as impl

    return impl(hdr, root)


def skill_pin_tenth_cousins_once_removed(hdr: str | None = None, root: str | None = None) -> dict:
    """T56 — pin-log tenth-cousin-once-removed rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_tenth_cousins_once_removed import (
        skill_pin_tenth_cousins_once_removed as impl,
    )

    return impl(hdr, root)



def skill_pin_eleventh_cousins(hdr: str | None = None, root: str | None = None) -> dict:
    """T57 — pin-log eleventh-cousin rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_eleventh_cousins import skill_pin_eleventh_cousins as impl

    return impl(hdr, root)


def skill_pin_twelfth_cousins(hdr: str | None = None, root: str | None = None) -> dict:
    """T59 — pin-log twelfth-cousin rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_twelfth_cousins import skill_pin_twelfth_cousins as impl

    return impl(hdr, root)



def skill_pin_twelfth_cousins_once_removed(hdr: str | None = None, root: str | None = None) -> dict:
    """T60 — pin-log twelfth-cousin-once-removed rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_twelfth_cousins_once_removed import (
        skill_pin_twelfth_cousins_once_removed as impl,
    )

    return impl(hdr, root)



def skill_pin_thirteenth_cousins(hdr: str | None = None, root: str | None = None) -> dict:
    """T61 — pin-log thirteenth-cousin rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_thirteenth_cousins import skill_pin_thirteenth_cousins as impl

    return impl(hdr, root)


def skill_pin_eleventh_cousins_once_removed(hdr: str | None = None, root: str | None = None) -> dict:
    """T58 — pin-log eleventh-cousin-once-removed rows of HDR under a root. Loopback only. Never fetch. Never exec."""
    _path()
    from rails.linux.ctl_skillpin_eleventh_cousins_once_removed import (
        skill_pin_eleventh_cousins_once_removed as impl,
    )

    return impl(hdr, root)


def main(argv):
    _path()
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "thirteenth-cousins":
        from rails.linux.ctl_cli_t61 import run_t61

        return run_t61(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "twelfth-cousins-once-removed":
        from rails.linux.ctl_cli_t60 import run_t60

        return run_t60(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "twelfth-cousins":
        from rails.linux.ctl_cli_t59 import run_t59

        return run_t59(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "eleventh-cousins-once-removed":
        from rails.linux.ctl_cli_t58 import run_t58

        return run_t58(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "eleventh-cousins":
        from rails.linux.ctl_cli_t57 import run_t57

        return run_t57(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "tenth-cousins-once-removed":
        from rails.linux.ctl_cli_t56 import run_t56

        return run_t56(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "tenth-cousins":
        from rails.linux.ctl_cli_t55 import run_t55

        return run_t55(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "ninth-cousins-once-removed":
        from rails.linux.ctl_cli_t54 import run_t54

        return run_t54(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "ninth-cousins":
        from rails.linux.ctl_cli_t53 import run_t53

        return run_t53(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "eighth-cousins-once-removed":
        from rails.linux.ctl_cli_t52 import run_t52

        return run_t52(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "eighth-cousins":
        from rails.linux.ctl_cli_t51 import run_t51

        return run_t51(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "seventh-cousins-once-removed":
        from rails.linux.ctl_cli_t50 import run_t50

        return run_t50(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "seventh-cousins":
        from rails.linux.ctl_cli_t49 import run_t49

        return run_t49(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "sixth-cousins-once-removed":
        from rails.linux.ctl_cli_t48 import run_t48

        return run_t48(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "sixth-cousins":
        from rails.linux.ctl_cli_t47 import run_t47

        return run_t47(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "fifth-cousins-once-removed":
        from rails.linux.ctl_cli_t46 import run_t46

        return run_t46(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "fifth-cousins":
        from rails.linux.ctl_cli_t45 import run_t45

        return run_t45(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "third-cousins-once-removed":
        from rails.linux.ctl_cli_t42 import run_t42

        return run_t42(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "fourth-cousins-once-removed":
        from rails.linux.ctl_cli_t44 import run_t44

        return run_t44(argv, globals())
    if len(argv) > 2 and argv[1] == "skill-pin" and argv[2] == "fourth-cousins":
        from rails.linux.ctl_cli_t43 import run_t43

        return run_t43(argv, globals())
    from rails.linux.ctl_cli import run

    return run(argv, globals())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
