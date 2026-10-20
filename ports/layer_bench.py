"""Sandbox bench for layered ternary compute. Local only."""
from __future__ import annotations
import json
import time
from pathlib import Path
from ports.layer_compute import layer_score

def bench(n: int = 1000) -> dict:
    notes = ["os tick", "goldend status note for envelope", "home rail", "web3node sample value"]
    t0 = time.perf_counter()
    results = []
    for i in range(n):
        results.append(layer_score(notes[i % len(notes)]))
    ms = (time.perf_counter() - t0) * 1000 / n
    layers = {}
    for r in results:
        layers[r["layer"]] = layers.get(r["layer"], 0) + 1
    body = {
        "n": n,
        "ms_per": round(ms, 4),
        "layers": layers,
        "sample": results[-1],
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "layer_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(bench(), indent=2))
