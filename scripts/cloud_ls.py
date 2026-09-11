#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.bitnet_cloud import rows


def main(argv: list[str]) -> int:
    vault = Path(argv[1]) if len(argv) > 1 else Path("/tmp/juniorhome_vault")
    data = rows(vault)
    print(json.dumps({"n": len(data), "areas": [r.get("area") for r in data], "local": all(r.get("local") for r in data) if data else True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
