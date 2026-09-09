from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.state_schema import receipt, validate, write_pair


class StateSchemaTests(unittest.TestCase):
    def test_validate(self):
        r = receipt("A9", "JuniorAstra", "shipped", "B2")
        self.assertEqual(validate(r), [])
        self.assertTrue(validate({"at": "x"}))

    def test_write(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_pair(root, receipt("A9", "JuniorAstra", "shipped", "B2"))
            self.assertTrue((root / "STATE.md").is_file())
            self.assertTrue((root / "grok_bot" / "LAST_RECEIPT.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
