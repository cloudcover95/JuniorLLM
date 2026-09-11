from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.flagstaff_balance import guess
from ports.use_in import ingest


class UseInTests(unittest.TestCase):
    def test_cases(self):
        self.assertEqual(guess("trim mp3 mixdown"), "audio")
        self.assertEqual(guess("note taking analysis journal"), "notes")
        self.assertEqual(guess("pyside gui app"), "app")
        self.assertEqual(guess("dxf omega drawing"), "cad")
        with tempfile.TemporaryDirectory() as td:
            a = ingest(Path(td), "trim mp3 mixdown")
            n = ingest(Path(td), "note taking analysis journal")
            self.assertEqual(a["inject"]["inbox"], "audio_inbox.jsonl")
            self.assertEqual(n["inject"]["inbox"], "notes_inbox.jsonl")
            self.assertTrue(a["tp_match"] and n["tp_match"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
