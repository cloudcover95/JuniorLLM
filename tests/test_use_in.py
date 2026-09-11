from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.flagstaff_balance import guess
from ports.use_in import ingest


class UseTests(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(guess("mp3 trim edit wav"), "media")
        self.assertEqual(guess("journal analysis expand"), "notes")
        self.assertEqual(guess("photocopy scan tiff store"), "scan")
        with tempfile.TemporaryDirectory() as td:
            for k in ("media", "notes", "scan"):
                r = ingest(Path(td), k)
                self.assertTrue(r["ok"], k)
                self.assertEqual(r["guess"], k)


if __name__ == "__main__":
    unittest.main(verbosity=2)
