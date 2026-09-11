from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junior_bitnet.prove import prove
from junior_bitnet.teqp import props


class TeqpTests(unittest.TestCase):
    def test_sparse_vs_dense(self):
        s = props([0] * 30 + [1, -1])
        d = props([1, -1] * 16)
        self.assertLess(s.rho, d.rho)
        self.assertEqual(s.phase, "sparse")
        self.assertEqual(d.phase, "dense")

    def test_prove(self):
        p = prove()
        self.assertTrue(p["ok"], p["hypotheses"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
