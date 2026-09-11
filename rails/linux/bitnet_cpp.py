"""Launch upstream bitnet.cpp if the binary and a GGUF exist. Else refuse.
Does not compile a kernel module. Does not patch AGX firmware.
"""
from __future__ import annotations

import os
from pathlib import Path

from rails.linux.backend import probe


def plan() -> dict:
    p = probe()
    gguf = os.environ.get("JUNIOR_GGUF")
    ready = bool(p.bitnet_cpp and gguf and Path(gguf).is_file())
    cmd = None
    if ready:
        cmd = [p.bitnet_cpp, "-m", gguf, "--host", "127.0.0.1", "--port", "8765"]
    return {
        "ready": ready,
        "cpp": p.bitnet_cpp,
        "gguf": gguf,
        "asahi": p.asahi,
        "accel": p.accel,
        "cmd": cmd,
        "fallback": "python3 rails/linux/bitnetd.sh",
    }
