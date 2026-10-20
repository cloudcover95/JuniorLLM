"""Goldend entry that runs the web3 full path. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.full_path_web3 import path

def run(note: str = "goldend web3") -> dict:
    result = path(note, use_web3=True)
    out = Path.home() / ".juniorhome" / "os" / "goldend_web3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
