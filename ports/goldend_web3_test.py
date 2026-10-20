"""Self-test for goldend web3. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.goldend_web3 import run

def test(note: str = "goldend test") -> dict:
    result = run(note)
    ok = result.get("check", {}).get("ok") is True and "ternary" in result
    body = {"ok": ok, "energy": result["check"].get("energy"), "cheaper": result["ternary"].get("cheaper"), "local": True}
    out = Path.home() / ".juniorhome" / "os" / "goldend_web3_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(test(), indent=2))
