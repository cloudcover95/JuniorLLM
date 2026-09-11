from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junior_bitnet.palace import Palace
from junior_bitnet.prove import prove
from lattice_zk.ternary_arg import verify


class PalaceTests(unittest.TestCase):
    def test_poison_copy(self):
        p = Palace()
        p.seal("a", [1, 0, -1, 1] * 8)
        z = p.pull("a")
        z[:] = [0] * len(z)
        self.assertTrue(verify(p.slots["a"].proof))
        self.assertNotEqual(p.slots["a"].z, z)

    def test_prove(self):
        out = prove()
        self.assertTrue(out["ok"], out["hypotheses"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
