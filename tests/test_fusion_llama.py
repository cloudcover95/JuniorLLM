from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junior_bitnet.bitlinear import bitlinear
from junior_bitnet.fusion import run
from rails.linux.llama import plan


class FusionLlamaTests(unittest.TestCase):
    def test_fused_matches_bitlinear(self):
        x = [0.5, -0.2, 0.1]
        w = [1.2, -0.4, 0.05]
        a = bitlinear(x, w)
        b = run(x, w)
        self.assertAlmostEqual(a["y"], b["y"])
        self.assertEqual(b["fused"], True)
        self.assertIn(b["backend"], {"cpu", "triton"})

    def test_llama_fail_closed(self):
        self.assertFalse(plan()["ready"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
