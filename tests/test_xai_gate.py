from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.xai_gate import complete, status


class XaiGateTests(unittest.TestCase):
    def test_no_call(self):
        s = status()
        self.assertTrue(s["paid_api"])
        self.assertFalse(s["unlimited_free"])
        self.assertFalse(s["call"])
        self.assertFalse(complete("hi")["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
