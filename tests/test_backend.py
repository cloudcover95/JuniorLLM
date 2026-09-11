from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rails.linux.backend import probe, trit


class BackendTests(unittest.TestCase):
    def test_alphabet(self):
        z = trit([1.0, -0.2, 0.01, 4.0])
        self.assertTrue(all(t in (-1, 0, 1) for t in z))

    def test_probe_fields(self):
        p = probe()
        self.assertIn(p.accel, {"mlx", "cuda", "cpu"})
        self.assertEqual(p.kernel, "junior_bitnet.absmean")


if __name__ == "__main__":
    unittest.main(verbosity=2)
