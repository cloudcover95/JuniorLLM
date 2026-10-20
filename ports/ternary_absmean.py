"""Ternary check that includes absmean energy. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.web3_absmean import feed
from ports.ternary_efficiency import check as trit_check

def check(note: str = "status note") -> dict:
    energy = feed(note)
    trit = trit_check(note=note)
    body = {
        "absmean": energy["value"],
        "trit_ms": trit["trit_ms"],
        "baseline_ms": trit["baseline_ms"],
        "cheaper": trit["cheaper"],
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "ternary_absmean.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
