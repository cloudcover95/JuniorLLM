from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports import flagstaff
from ports.flagstaff import assemble


class CacheTests(unittest.TestCase):
    def test_second_ask_hits(self):
        flagstaff.CACHE.store.clear()
        flagstaff.CACHE.hits = flagstaff.CACHE.miss = 0
        a = assemble("flagstaff dry beta")
        b = assemble("flagstaff dry beta")
        self.assertEqual(a["cache"], "miss")
        self.assertEqual(b["cache"], "hit")
        self.assertEqual(a["port"], "JuniorBitNetFieldCore")
        self.assertGreaterEqual(flagstaff.CACHE.hits, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
