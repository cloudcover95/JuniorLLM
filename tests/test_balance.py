from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.flagstaff_balance import check
from ports.inject import digest


class BalanceTests(unittest.TestCase):
    def test_pass(self):
        b = check("flagstaff dry beta")
        self.assertTrue(b["ok"])
        self.assertTrue(all(b["votes"].values()))

    def test_private(self):
        self.assertFalse(digest("x", private=True, consent=False)["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
