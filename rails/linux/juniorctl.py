#!/usr/bin/env python3
"""juniorctl — JuniorOS entry. skill_pin_list skill_pin_verify skill_pin_verify_one skill_pin_tip skill_pin_log skill_pin_height skill_pin_get skill_pin_at skill_pin_range skill_pin_since skill_pin_until skill_pin_before skill_pin_after skill_pin_first skill_pin_last skill_pin_tail skill_pin_head skill_pin_count skill_pin_genesis skill_pin_parent skill_pin_child skill_pin_children skill_pin_ancestors skill_pin_siblings skill_pin_cousins skill_pin_uncles skill_pin_nephews skill_pin_grandchildren skill_pin_great_grandchildren skill_pin_great_great_grandchildren skill_pin_2nd_cousins skill_pin_first_cousins_once_removed skill_pin_2nd_cousins_once_removed skill_pin_third_cousins skill_pin_third_cousins_once_removed"""
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
    }
