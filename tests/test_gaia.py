from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.gaia import GAIA, spine


class GaiaTests(unittest.TestCase):
    def test_no_pull(self):
        s = spine("gaia they home")
        self.assertEqual(GAIA.max_download_gb, 0.0)
        self.assertFalse(s["download"])
        self.assertFalse(s["ue5_launch"])
        self.assertTrue(s["ok"])
        self.assertEqual(s["who"]["name"], "Gaia")


if __name__ == "__main__":
    unittest.main(verbosity=2)
