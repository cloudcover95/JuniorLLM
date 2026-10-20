"""Self-test for the full path. Local only. No model pull."""
from __future__ import annotations
import json
from pathlib import Path
from ports.full_path import path

def test() -> dict:
    short = path("os")
    long_note = path("x" * 200)
    ok = short["layers"]["layer"] in ("T0", "T1", "T2")
    ok = ok and "decision" in short and "record" in short
    body = {
        "short_layer": short["layers"]["layer"],
        "long_layer": long_note["layers"]["layer"],
        "ok": ok,
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "full_path_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(test(), indent=2))
