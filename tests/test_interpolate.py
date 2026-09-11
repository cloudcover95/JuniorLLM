from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptations.omega_cad.draft import interpret
from adaptations.omega_cad.interpolate import interpolate


class InterpTests(unittest.TestCase):
    def test_explicit(self):
        i = interpolate("40 20", "", "HEIGHT: 8")
        self.assertEqual(i.height, 8.0)
        self.assertFalse(i.hypothesis)

    def test_misc_median(self):
        i = interpolate("profile 40 20 outline", "THK 6 note 6 dim 6", "TITLE BRACKET")
        self.assertIsNotNone(i.height)
        self.assertTrue(i.hypothesis)
        self.assertIn(i.source, {"misc_median", "profile_quarter", "blocked"})

    def test_draft_does_not_extrude_on_guess(self):
        c = interpret("TITLE: BRACKET", profile="40 20", misc="6 6 6")
        self.assertTrue(c.interp_hypothesis)
        self.assertFalse(c.allow_extrude)


if __name__ == "__main__":
    unittest.main(verbosity=2)
