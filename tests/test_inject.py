from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.flagstaff_balance import guess
from ports.inject import digest, write_vault


class InjectTests(unittest.TestCase):
    def test_guess(self):
        self.assertEqual(guess("dxf missing height"), "cad")
        self.assertEqual(guess("custom node book"), "stock")
        self.assertEqual(guess("flagstaff dry V4 crimp"), "climbs")
        self.assertEqual(guess("asahi mlx overlay"), "asahi")
        self.assertEqual(guess("triton cuda fused dot"), "cuda")
        self.assertEqual(guess("aarch64 pi i2sd"), "arm")

    def test_consent(self):
        self.assertFalse(digest("x", private=True, consent=False)["ok"])
        row = digest("triton cuda fused dot")
        self.assertTrue(row["ok"])
        self.assertEqual(row["area"], "cuda")
        self.assertEqual(row["inbox"], "kernel_inbox.jsonl")

    def test_vault(self):
        with tempfile.TemporaryDirectory() as td:
            write_vault("dxf missing height", Path(td))
            self.assertTrue((Path(td) / "cad_inbox.jsonl").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
