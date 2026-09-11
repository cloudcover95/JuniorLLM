#!/usr/bin/env python3
"""juniorctl — JuniorOS entry."""
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


def night(ticks: int = 32) -> dict:
    _path()
    from bitnet_night.cycle import run_cycle

    return run_cycle(ROOT / "agent" / "queue", "complete local suite and JuniorOS overlay", ticks=ticks)


def ports() -> list:
    _path()
    from ports.registry import list_ports

    return list_ports()


def quant() -> dict:
    _path()
    from scripts.bitnet_quant_prod import run

    return run()


def lake() -> dict:
    _path()
    import tempfile
    from bitnet_pq.pipeline import run as lake_run

    return lake_run(Path(tempfile.mkdtemp(prefix="juniorctl-lake-")))


def net_status() -> dict:
    _path()
    import tempfile
    from bitnet_net.node import Node

    n = Node(Path(tempfile.mkdtemp(prefix="juniorctl-net-")))
    n.mint("ctl", 1)
    n.seal()
    return {"balances": n.balances, "height": len(n.blocks), "bind": "127.0.0.1"}


def oci_validate() -> dict:
    """C5 — juniorctl wrapper around the C4 rootless OCI validator."""
    _path()
    from adaptations.gemma4.ondisk_bind import notes as gemma_notes
    from rails.linux.oci import rootless

    report = rootless.validate()
    unit = rootless.unit()
    gemma = gemma_notes()
    return {
        "ok": bool(report["ok"]),
        "issues": list(report["issues"]),
        "bind": report["bind"],
        "rootless": bool(report["rootless"]),
        "privileged": False,
        "docker_socket": False,
        "unit": unit["name"],
        "unit_status": unit["status"],
        "gemma": {
            "port": gemma["port"],
            "present": bool(gemma["present"]),
            "path": gemma.get("path"),
            "backend": gemma["backend"],
            "fetch": False,
        },
        "weights": unit.get("weights"),
    }


def oci_install(dest: str | None = None) -> dict:
    """C6 — stage the C4 OCI bundle under DEST. Loopback only. No fetch."""
    import os

    _path()
    from rails.linux.oci.install_bundle import stage

    target = dest if dest else os.environ.get("DEST", "")
    return stage(target)


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "health"
    if cmd == "health":
        print(json.dumps(health(), indent=2))
        return 0
    if cmd == "security":
        report = security()
        print(json.dumps(report, indent=2))
        return 0 if report["unit_ok"] and report["seccomp"] else 1
    if cmd == "port" and len(argv) > 2 and argv[2] == "list":
        print(json.dumps(ports(), indent=2))
        return 0
    if cmd == "ask":
        q = " ".join(argv[2:]).strip() or "public field conditions brief"
        print(json.dumps(ask(q), indent=2))
        return 0
    if cmd == "night":
        print(json.dumps(night(), indent=2))
        return 0
    if cmd == "quant":
        print(json.dumps(quant(), indent=2, default=str))
        return 0
    if cmd == "lake":
        print(json.dumps(lake(), indent=2))
        return 0
    if cmd == "net":
        print(json.dumps(net_status(), indent=2))
        return 0
    if cmd == "oci":
        sub = argv[2] if len(argv) > 2 else "validate"
        if sub == "validate":
            report = oci_validate()
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "install":
            dest = argv[3] if len(argv) > 3 else None
            report = oci_install(dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        print("usage: juniorctl oci validate | oci install [DEST]", file=sys.stderr)
        return 2
    print(
        "usage: juniorctl health | security | port list | ask <q> | night | quant | lake | net | oci validate | oci install [DEST]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
