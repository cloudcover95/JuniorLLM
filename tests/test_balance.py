from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.flagstaff_balance import check
from ports.inject import digest


class BalanceTests(unittest.TestCase):
    def test_home(self):
        self.assertTrue(check("local llm card", area="home")["ok"])

    def test_cad(self):
        b = check("dxf missing height", area="cad")
        self.assertTrue(b["ok"])
        self.assertEqual(b["port"], "JuniorBitNetDraft")

    def test_stock(self):
        self.assertTrue(check("custom node book", area="stock")["ok"])

    def test_private(self):
        self.assertFalse(digest("x", private=True, consent=False)["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
