"""Self-test for the goldend web3 path. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.goldend_web3 import run

def test() -> dict:
    result = run("os")
    ok = "layers" in result and "decision" in result and result.get("web3") is True
    body = {
        "ok": ok,
        "layer": result.get("layers", {}).get("layer"),
        "keep": result.get("decision", {}).get("keep"),
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "goldend_web3_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(test(), indent=2))
