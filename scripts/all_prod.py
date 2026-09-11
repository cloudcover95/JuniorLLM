#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rails.linux.backend import report
from rails.linux.llama import plan
from scripts.stack_prod import main as stack


def main(argv: list[str]) -> int:
    rc = stack(argv)
    extra = {"probe": report(), "llama": {"ready": plan()["ready"]}}
    print(json.dumps(extra, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
