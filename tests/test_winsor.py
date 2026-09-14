from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junior_bitnet.winsor import pack


class WinsorTests(unittest.TestCase):
    def test_trit_set(self):
        r = pack([0.01, -2.0, 0.4, 9.0, -0.02])
        self.assertTrue(r["ok"])
        self.assertTrue(set(r["trit"]) <= {-1, 0, 1})
        self.assertGreater(r["gamma"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
