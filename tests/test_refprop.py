from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junior_bitnet.palace import Palace
from junior_bitnet.refprop import Library
from junior_bitnet.vault import write_vault
from lattice_zk.ternary_arg import verify


class RefpropTests(unittest.TestCase):
    def test_catalog(self):
        lib = Library()
        self.assertIn("NIGHT", lib.names())
        self.assertGreater(float(lib.props_si("D", "DENSE")), float(lib.props_si("D", "SPARSE")))
        self.assertTrue(lib.rows["NIGHT"].sealed)
        self.assertTrue(verify(lib.palace.slots["night"].proof))

    def test_vault(self):
        with tempfile.TemporaryDirectory() as td:
            p = write_vault(Path(td))
            text = p.read_text(encoding="utf-8")
            self.assertIn("JuniorTeqp", text)
            self.assertIn("NIGHT", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
