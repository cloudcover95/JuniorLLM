"""Full-path sandbox: layer score, terraform decide, inference record. Local only."""
from __future__ import annotations
import json
import time
from pathlib import Path
from ports.layer_compute import layer_score
from ports.osai_terraform import decide
from ports.inference_engine import record

def path(note: str) -> dict:
    t0 = time.perf_counter()
    layers = layer_score(note)
    decision = decide(note)
    rec = record(note, source="full_path")
    ms = (time.perf_counter() - t0) * 1000
    body = {
        "layers": layers,
        "decision": decision,
        "record": rec,
        "ms": round(ms, 4),
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "full_path.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(path("os tick"), indent=2))
