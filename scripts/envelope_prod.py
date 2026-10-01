#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.osai_envelope import run

if __name__ == "__main__":
    env = sys.argv[1] if len(sys.argv) > 1 else "host"
    task = sys.argv[2] if len(sys.argv) > 2 else "JuniorOSai"
    print(json.dumps(run(task, env), indent=2))
