"""juniorctl overlay ops. Loopback only. Never fetch. Never exec."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path() -> None:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


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
    import os

    _path()
    from rails.linux.oci.install_bundle import stage

    target = dest if dest else os.environ.get("DEST", "")
    return stage(target)


def path_pin(dest: str | None = None) -> dict:
    import os

    _path()
    from rails.linux.path_pin import stage

    target = dest if dest else os.environ.get("DEST", "")
    return stage(target)
