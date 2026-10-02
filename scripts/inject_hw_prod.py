#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.inject_hw import inject

if __name__ == "__main__":
    hw = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    print(json.dumps(inject("JuniorOS", hw), indent=2))
