#!/usr/bin/env python3
"""List compiled-backlog skeleton paths that are still missing."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [
    "docs/COMPILED_BACKLOG.md",
    "docs/BETA_TO_OS.md",
    "docs/TEST_ROADMAP_2026.md",
    "STATE.md",
    "evals/contracts.py",
    "rails/linux/install-overlay.sh",
    "rails/linux/CONTAINER_SECURITY.md",
    "rails/swift/Sources/JuniorRails/Guardrail.swift",
    "junior_aie/framework.py",
    "ports/ON_DEVICE.md",
    "grok_bot/LAST_RECEIPT.md",
    "adaptations/multi_portal_production_loop.py",
]


def main() -> int:
    missing = [p for p in EXPECTED if not (ROOT / p).is_file()]
    print("missing", missing or "none")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
