from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptations.omega_cad.interpolate import interpolate
from adaptations.omega_cad.quant import absmean, i2s_pack
from scripts.quant_bench import main as bench


class QuantMethodTests(unittest.TestCase):
    def test_absmean(self):
        t, sc = absmean([40.0, 20.0, 8.0, 0.2])
        self.assertTrue(all(x in (-1, 0, 1) for x in t))
        self.assertGreater(sc, 0)
        self.assertTrue(i2s_pack(t))

    def test_explicit_skips_block(self):
        i = interpolate("40 20", "999", "HEIGHT: 8")
        self.assertEqual(i.source, "explicit")
        self.assertFalse(i.hypothesis)
        self.assertEqual(i.height, 8.0)

    def test_bench_runs(self):
        out = bench()
        self.assertLess(out["absmean_us"], 5000)
        self.assertLess(out["interp_explicit_us"], out["interp_misc_us"] * 8 + 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
