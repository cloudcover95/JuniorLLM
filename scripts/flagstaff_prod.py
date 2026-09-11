#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.flagstaff import assemble
from ports.layer1_iq import cycle


def main() -> int:
    lock = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/layer1_lock.json")
    cycle(lock, "flagstaff dry beta")
    print(json.dumps(assemble("flagstaff dry beta", lock), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
