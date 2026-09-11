from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rails.linux.backend import probe
from rails.linux.bitnet_cpp import plan


class CppTests(unittest.TestCase):
    def test_not_ready_without_gguf(self):
        p = plan()
        self.assertFalse(p["ready"])
        self.assertIn(probe().accel, {"mlx", "cuda", "cpu"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
