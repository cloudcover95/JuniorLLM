from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptations.omega_cad.draft import interpret
from ports.registry import pick


class DraftPortTests(unittest.TestCase):
    def test_pick(self):
        self.assertEqual(pick("dxf title block", 4).name, "JuniorBitNetDraft")
        self.assertEqual(pick("cad drawing", 16).name, "JuniorBitNetDraft")

    def test_interpret(self):
        c = interpret("TITLE: BRACKET\nELEV FRONT\nTHICKNESS: 8\n")
        self.assertEqual(c.port, "JuniorBitNetDraft")
        self.assertEqual(c.height, 8.0)
        self.assertIn(c.rec, {"high_confidence", "review_needed", "low_confidence"})

    def test_no_elev_blocks_extrude(self):
        c = interpret("TITLE: mystery scan")
        self.assertFalse(c.allow_extrude)


if __name__ == "__main__":
    unittest.main(verbosity=2)
