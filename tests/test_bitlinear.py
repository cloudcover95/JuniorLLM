from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junior_bitnet.bitlinear import bitlinear
from junior_bitnet.math import binarize
from junior_bitnet.prove import prove


class BitlinearTests(unittest.TestCase):
    def test_binarize(self):
        self.assertEqual(binarize([-2.0, 0.0, 3.0]), [-1, 1, 1])

    def test_bitlinear(self):
        y = bitlinear([1.0, -1.0, 0.5], [0.2, -0.2, 0.01])
        self.assertTrue(all(w in (-1, 0, 1) for w in y["wq"]))

    def test_prove_ecosystem(self):
        p = prove()
        self.assertTrue(p["ok"], p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
