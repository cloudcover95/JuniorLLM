#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.surface_infer import infer

if __name__ == "__main__":
    print(json.dumps(infer(sys.argv[1] if len(sys.argv) > 1 else "web"), indent=2))
