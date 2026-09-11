from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.app_in import ingest
from ports.flagstaff_balance import guess


class AppInTests(unittest.TestCase):
    def test_app_not_cad(self):
        self.assertEqual(guess("pyside gui app feature flag"), "app")
        self.assertEqual(guess("dxf omega drawing"), "cad")
        with tempfile.TemporaryDirectory() as td:
            out = ingest(Path(td))
            self.assertTrue(out["balance"])
            self.assertEqual(out["guess"], "app")
            self.assertEqual(out["inject"]["inbox"], "app_inbox.jsonl")
            self.assertTrue(out["tp_match"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
