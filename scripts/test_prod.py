#!/usr/bin/env python3
"""Local gate. No .github workflow."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SUITES = (
    "tests.test_inject",
    "tests.test_balance",
    "tests.test_bitnet_cloud",
    "tests.test_fusion_llama",
    "tests.test_enduser_llm",
    "tests.test_tp",
)


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in SUITES:
        suite.addTests(loader.loadTestsFromName(name))
    r = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if r.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
