"""Goldend full-path entry. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.full_path import path

def run(note: str = "goldend full path") -> dict:
    result = path(note)
    out = Path.home() / ".juniorhome" / "os" / "goldend_full_path.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
