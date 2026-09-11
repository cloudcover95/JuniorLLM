from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.bitnet_cloud import put, rows
from ports.inject import write_vault


class CloudTests(unittest.TestCase):
    def test_local(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            row = put(p, "triton cuda fused dot", "cuda")
            self.assertTrue(row["local"])
            self.assertFalse(row["cloud"])
            write_vault("dxf missing height", p)
            idx = (p / "bitnetCloud" / "index.jsonl").read_text()
            self.assertIn("cad", idx)
            self.assertTrue(all(r["local"] for r in rows(p)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
