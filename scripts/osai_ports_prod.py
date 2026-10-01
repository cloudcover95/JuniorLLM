#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.agi_sdk import card as agi
from ports.registry import pick
from ports.stonefield import list_public
from ports.web3node_port import card as web3

if __name__ == "__main__":
    print(json.dumps({
        "omega": pick("omega mesh", 0).name,
        "agi": agi(),
        "stonefield": len(list_public()),
        "web3node": web3(),
        "model_pull": False,
    }, indent=2))
