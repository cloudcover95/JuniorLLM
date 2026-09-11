#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.inject import write_vault


def main(argv: list[str]) -> int:
    vault = Path(argv[1]) if len(argv) > 1 else Path("/tmp/juniorhome_vault")
    note = argv[2] if len(argv) > 2 else "flagstaff dry beta"
    print(json.dumps(write_vault(note, vault), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
