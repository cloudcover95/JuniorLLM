#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.user_in import ingest


def main(argv: list[str]) -> int:
    vault = Path(argv[1]) if len(argv) > 1 else Path("/tmp/juniorhome_vault")
    note = argv[2] if len(argv) > 2 else "buy oats and fix the porch light"
    out = ingest(vault, note)
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
