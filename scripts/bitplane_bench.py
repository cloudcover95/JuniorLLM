#!/usr/bin/env python3
"""Bench bit-plane kernel on synthetic ternary weights. No GGUF. No model pull."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from junior_osai.kernels.bitplane import bench

if __name__ == "__main__":
    print(json.dumps(bench(512, 512, repeats=5), indent=2))
