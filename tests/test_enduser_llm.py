from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.enduser_llm import build, spec


class EndUserTests(unittest.TestCase):
    def test_card(self):
        s = spec()
        self.assertEqual(s["port"], "JuniorBitNetFieldCore")
        self.assertIn(s["runtime"], {"mlx", "llama.cpp", "i2sd"})
        with tempfile.TemporaryDirectory() as td:
            card = build(Path(td), "flagstaff home box")
            self.assertTrue((Path(td) / "local_llm.json").is_file())
            self.assertTrue(card["tf"]["ok"])
            disk = json.loads((Path(td) / "local_llm.json").read_text())
            self.assertEqual(disk["name"], s["name"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
