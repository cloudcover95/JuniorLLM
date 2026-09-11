from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junior_bitnet.coolstore import build, dump, load, props_si
from junior_bitnet.refprop import Library
from junior_bitnet.teqp import T_C


class CoolstoreTests(unittest.TestCase):
    def test_matches_live(self):
        lib = Library()
        tab = build(lib)
        self.assertNotIn("z", json_blob := __import__("json").dumps(tab))
        live = float(lib.props_si("D", "NIGHT"))
        cold = float(props_si(tab, "D", "NIGHT", T_C))
        self.assertAlmostEqual(live, cold, places=6)

    def test_roundtrip_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = dump(Path(td) / "coolstore.json")
            tab = load(p)
            self.assertEqual(tab["kind"], "junior-coolstore-v0")
            self.assertIn("NIGHT", tab["fluids"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
