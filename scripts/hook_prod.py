#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rails.linux.hook import HOST, PORT, serve


def main() -> int:
    print(f"hook {HOST}:{PORT}")
    serve().serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
