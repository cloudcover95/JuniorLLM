from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junior_bitnet.bitlinear import bitlinear
from junior_bitnet.tp import tp_bitlinear


class TPTests(unittest.TestCase):
    def test_match(self):
        x = [0.5, -0.2, 0.1, 0.8, -0.3]
        w = [1.2, -0.4, 0.05, -0.9, 0.3]
        a = bitlinear(x, w)
        for k in (1, 2, 4):
            b = tp_bitlinear(x, w, k)
            self.assertTrue(b["match"])
            self.assertAlmostEqual(a["y"], b["y"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
