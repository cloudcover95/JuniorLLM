from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.osai_goldens import run


class OsaiGoldensTests(unittest.TestCase):
    def test_birds(self):
        r = run()
        self.assertEqual(r["species"], "birds")
        self.assertFalse(r["download"])
        self.assertEqual(r["passed"], r["n"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
