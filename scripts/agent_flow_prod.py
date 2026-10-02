#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.agent_flow import step

if __name__ == "__main__":
    print(json.dumps(step(), indent=2))
