from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.omega_in import ingest


class OmegaInTests(unittest.TestCase):
    def test_sample(self):
        with tempfile.TemporaryDirectory() as td:
            out = ingest(Path(td))
            self.assertTrue(out["balance"])
            self.assertTrue(out["tp_match"])
            self.assertTrue(out["inject"]["ok"])
            self.assertEqual(out["guess"], "cad")
            self.assertEqual(out["port"], "JuniorBitNetDraft")
            self.assertFalse(out["llm"]["llama_ready"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
