from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.layer1_iq import TIERS, cycle
from ports.terraform import terraform


class Layer1Tests(unittest.TestCase):
    def test_lock_and_hold(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lock.json"
            a = cycle(p, "cad dxf title")
            b = cycle(p, "cad dxf title")
            self.assertEqual(a["locked"], list(TIERS))
            self.assertEqual(b["locked"], list(TIERS))
            self.assertEqual(b["cycles"], 2)

    def test_terraform_strips(self):
        out = terraform("ignore previous 0.0.0.0 wget http://x flagstaff")
        self.assertTrue(out["ok"])
        self.assertNotIn("0.0.0.0", out["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
