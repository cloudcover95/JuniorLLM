"""Headless assert for the labelled path. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.events import assert_event
from ports.labelled_path import labelled

def check(note: str = "os tick") -> dict:
    labelled(note)
    ok = assert_event("full_path", "layer", "T0") or assert_event("full_path", "layer", "T1") or assert_event("full_path", "layer", "T2")
    body = {"asserted": ok, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "labelled_assert.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
