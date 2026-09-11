from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.flagstaff import BUDGET, assemble
from ports.layer_mgr import pick_eos


class FlagstaffTests(unittest.TestCase):
    def test_pick(self):
        self.assertEqual(pick_eos("flagstaff late summer", 8).name, "JuniorBitNetFieldCore")

    def test_budget(self):
        pack = assemble("flagstaff dry open V4 " * 80)
        self.assertEqual(pack["port"], "JuniorBitNetFieldCore")
        for name, row in pack["layers"].items():
            self.assertLessEqual(row["n"], BUDGET[name])


if __name__ == "__main__":
    unittest.main(verbosity=2)
