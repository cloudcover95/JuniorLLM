from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.ondisk import probe, ready


class OnDiskTests(unittest.TestCase):
    def test_missing_is_false(self):
        with tempfile.TemporaryDirectory() as td:
            rows = probe(Path(td))
            self.assertTrue(all(not r.present for r in rows))
            self.assertFalse(ready("BitNet-2B4T", Path(td)))

    def test_present(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bitnet-b1.58-2B-4T-I2_S.gguf"
            p.write_bytes(b"")
            self.assertTrue(ready("BitNet-2B4T", Path(td)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
