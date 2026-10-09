"""T59 CLI dispatch for skill-pin twelfth-cousins. Loopback only."""
from __future__ import annotations

import json


def run_t59(argv: list[str], ns: dict) -> int:
    hdr = argv[3] if len(argv) > 3 else None
    dest = argv[4] if len(argv) > 4 else None
    report = ns["skill_pin_twelfth_cousins"](hdr, dest)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1
