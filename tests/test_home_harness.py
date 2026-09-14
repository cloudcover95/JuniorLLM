from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.home_harness import harness


class HomeHarnessTests(unittest.TestCase):
    def test_no_launch_no_download(self):
        h = harness("ue5 spark", 90)
        self.assertFalse(h["ue5_launch"])
        self.assertFalse(h["download"])
        self.assertFalse(h["kernel_patch"])
        self.assertFalse(h["xai_call"])
        self.assertEqual(h["bind"]["hook"], "127.0.0.1:8770")

    def test_fieldcore_port(self):
        h = harness("flagstaff fieldcore")
        self.assertEqual(h["surface"]["port"], "JuniorBitNetFieldCore")


if __name__ == "__main__":
    unittest.main(verbosity=2)
