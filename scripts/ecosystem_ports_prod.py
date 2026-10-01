#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.ecosystem import list_cores, route

if __name__ == "__main__":
    print(json.dumps({"n": len(list_cores()), "cores": list_cores(), "sample": route("JuniorEngrTools calc").name}, indent=2))
