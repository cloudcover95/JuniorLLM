#!/usr/bin/env python3
"""juniorctl — future JuniorOS entry. Stdlib CLI stub."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def health() -> dict:
    return {
        "product": "JuniorOS overlay",
        "bitnetd": "127.0.0.1:8765",
        "controls": str(ROOT / "grok_bot" / "CONTROLS.md"),
        "one_task": "local software + BitNet Linux OS roadmap",
    }


def ports() -> list:
    sys.path.insert(0, str(ROOT))
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
    print("usage: juniorctl health | port list", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
