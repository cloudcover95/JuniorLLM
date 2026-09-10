#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from bitnet_pq.pipeline import run


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(tempfile.mkdtemp(prefix="junior-lake-"))
    print(json.dumps(run(root), indent=2))
    print("root", root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
