from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports import terraform as tf
from ports.terraform import batch


class BatchTests(unittest.TestCase):
    def test_batch_cache(self):
        tf._Y.clear()
        rows = batch(["flagstaff dry beta", "flagstaff dry beta", "xanadu crimp"])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["fusion_y"], rows[1]["fusion_y"])
        self.assertEqual(len(tf._Y), 2)
        self.assertTrue(all(r["ok"] for r in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
