from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.inject import digest, write_vault


class InjectTests(unittest.TestCase):
    def test_consent(self):
        self.assertFalse(digest("x", private=True, consent=False)["ok"])
        row = digest("flagstaff dry V4 crimp")
        self.assertTrue(row["ok"])
        self.assertEqual(row["stone"]["port"], "JuniorBitNetFieldCore")
        self.assertEqual(row["stock"]["kind"], "custom")

    def test_vault(self):
        with tempfile.TemporaryDirectory() as td:
            write_vault("flagstaff slab", Path(td))
            self.assertTrue((Path(td) / "stonefield_inbox.jsonl").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
