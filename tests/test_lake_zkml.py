from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bitnet_pq.pipeline import run
from bitnet_pq.security import audit
from bitnet_pq.zkml import prove_forward, verify_forward


class LakeZkmlTests(unittest.TestCase):
    def test_zkml(self):
        rec = prove_forward([1, -1, 0, 1] * 8, layers=2)
        self.assertTrue(verify_forward(rec))
        self.assertTrue(audit(rec.proof).ok)
        self.assertFalse(rec.proof.secure)

    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            a = run(Path(td))
            b = run(Path(td))
            self.assertTrue(a["ledger_ok"] and b["ledger_ok"])
            self.assertEqual(b["height"], a["height"] + 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
