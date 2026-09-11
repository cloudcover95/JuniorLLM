from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.bitnet_cloud import rows
from scripts.stack_prod import NOTES, main


class StackTests(unittest.TestCase):
    def test_pass(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(main(["stack", td]), 0)
            self.assertGreaterEqual(len(rows(Path(td))), len(NOTES))
            self.assertTrue((Path(td) / "local_llm.json").is_file())
            self.assertTrue((Path(td) / "bitnetCloud" / "README.md").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
