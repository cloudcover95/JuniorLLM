#!/usr/bin/env python3
"""juniorctl — JuniorOS / end-user entry."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def health() -> dict:
    _path()
    from ports.registry import list_ports

    return {
        "product": "JuniorOS overlay",
        "bitnetd": "127.0.0.1:8765",
        "aie": "junior_aie on JuniorLLM main",
        "ports": [p["name"] for p in list_ports()],
        "ask": "juniorctl ask \"<question>\"",
        "one_task": "local software + BitNet Linux OS overlay",
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


def ports() -> list:
    _path()
    from ports.registry import list_ports

    return list_ports()


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "health"
    if cmd == "health":
        print(json.dumps(health(), indent=2))
        return 0
    if cmd == "port" and len(argv) > 2 and argv[2] == "list":
        print(json.dumps(ports(), indent=2))
        return 0
    if cmd == "ask":
        q = " ".join(argv[2:]).strip() or "public field conditions brief"
        print(json.dumps(ask(q), indent=2))
        return 0
    print("usage: juniorctl health | port list | ask <question>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
