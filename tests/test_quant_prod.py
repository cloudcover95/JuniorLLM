from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bitnet_quant_prod import run


class QuantProdTests(unittest.TestCase):
    def test_suite(self):
        out = run()
        self.assertFalse(out["secure"])
        self.assertTrue(out["lake"]["ledger_ok"])
        self.assertEqual(out["net"]["balances"]["member"], 2)
        self.assertTrue(out["pq_bench"]["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
