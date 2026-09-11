#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.omega_draft import run

SAMPLE = """TITLE: BRACKET
REV A
MM
VIEW: plan, elev
HEIGHT: 8
THICKNESS: 8
DIM_OK: yes
ELEV FRONT
"""


def main() -> int:
    print(json.dumps(run(SAMPLE), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
