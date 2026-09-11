from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junior_bitnet.i2s_c import matches


class CPackTests(unittest.TestCase):
    def test_match(self):
        self.assertTrue(matches([1, 0, -1, 1, 0, 1, -1, 0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
