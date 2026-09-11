from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.flagstaff_balance import guess
from ports.user_in import ingest


class UserInTests(unittest.TestCase):
    def test_anything(self):
        self.assertEqual(guess("buy oats and fix the porch light"), "user")
        self.assertEqual(guess("dxf missing height"), "cad")
        with tempfile.TemporaryDirectory() as td:
            r = ingest(Path(td), "buy oats and fix the porch light")
            self.assertTrue(r["ok"])
            self.assertEqual(r["guess"], "user")
            self.assertTrue(r["tp_match"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
