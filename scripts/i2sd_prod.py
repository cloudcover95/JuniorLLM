#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rails.linux.i2sd import serve


def main() -> int:
    print("i2sd 127.0.0.1:8767")
    serve().serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
