#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.terraform_lean import run

if __name__ == "__main__":
    surface = sys.argv[1] if len(sys.argv) > 1 else "code"
    print(json.dumps(run("JuniorOSai", surface), indent=2))
