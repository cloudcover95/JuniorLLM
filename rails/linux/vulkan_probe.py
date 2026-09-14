"""Vulkan is an operator box. Home does not dispatch compute shaders."""
from __future__ import annotations

import shutil
from pathlib import Path


def probe() -> dict:
    so = any(Path(p).is_file() for p in ("/usr/lib/libvulkan.so.1", "/usr/lib64/libvulkan.so.1"))
    return {
        "vulkaninfo": bool(shutil.which("vulkaninfo")),
        "libvulkan": so,
        "dispatch": False,
        "trit_on_gpu": False,
        "reason": "pack stays CPU winsor; GPU is a later box",
    }
