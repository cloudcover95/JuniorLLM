#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.stack_ports import run

if __name__ == "__main__":
    tasks = ["omega mesh", "agi sdk", "stonefield boulder", "web3node absmean", "osai"]
    print(json.dumps([run(t) for t in tasks], indent=2))
