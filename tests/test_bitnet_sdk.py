from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptations.omega_cad.draft import interpret
from junior_bitnet import compile_sheet, prove


class BitnetSdkTests(unittest.TestCase):
    def test_prove(self):
        p = prove()
        self.assertTrue(p["ok"], p)

    def test_compile_blocks(self):
        c = compile_sheet("notes only")
        self.assertFalse(c.ready)
        self.assertFalse(c.data["run_iq"])
        kinds = {a.kind + ":" + a.field for a in c.actions}
        self.assertIn("fix:height", kinds)

    def test_interpret_lists_actions(self):
        d = interpret("TITLE: x")
        self.assertFalse(d.allow_extrude)
        self.assertTrue(d.actions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
