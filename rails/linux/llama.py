"""llama.cpp / bitnet.cpp share the same GGUF loopback plan."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from rails.linux.backend import probe


def plan() -> dict:
    p = probe()
    bin_ = os.environ.get("JUNIOR_LLAMA") or shutil.which("llama-cli") or p.bitnet_cpp
    gguf = os.environ.get("JUNIOR_GGUF")
    ready = bool(bin_ and gguf and Path(str(gguf)).is_file() and Path(str(bin_)).exists())
    cmd = None
    if ready:
        cmd = [str(bin_), "-m", str(gguf), "--host", "127.0.0.1", "--port", "8765"]
    return {
        "ready": ready,
        "bin": bin_,
        "gguf": gguf,
        "accel": p.accel,
        "asahi": p.asahi,
        "cmd": cmd,
        "fallback": "i2sd",
    }
