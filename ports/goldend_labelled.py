"""Goldend labelled path. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.labelled_path import labelled
from ports.labelled_assert import check

def run(note: str = "goldend labelled") -> dict:
    path = labelled(note)
    asserted = check(note)
    body = {"path": path, "asserted": asserted["asserted"], "local": True}
    out = Path.home() / ".juniorhome" / "os" / "goldend_labelled.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
