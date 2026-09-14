"""Vulkan compute probe. Does not dispatch shaders."""
from __future__ import annotations

import shutil


def plan() -> dict:
    exe = shutil.which("vulkaninfo") or shutil.which("vkcube")
    return {
        "ready": bool(exe),
        "bin": exe,
        "dispatch": False,
        "spirv": False,
        "note": "Honeykrisp/Asahi and llama.cpp Vulkan are host stacks; Home does not ship GLSL",
    }
