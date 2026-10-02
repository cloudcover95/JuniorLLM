#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.os_infra import probe

if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))
