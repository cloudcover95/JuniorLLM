"""A8 — evals.contracts is the CI vocab source. Stdlib only."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.contracts import RECS, flops_proxy, rec_ok
from junior_aie.evalh import ci_gate, grade_trajectory, rec_vocab_grade


class RecVocab(unittest.TestCase):
    def test_known_labels(self):
        for label in ("high_confidence", "review_needed", "low_confidence"):
            self.assertTrue(rec_ok(label))
            self.assertTrue(rec_vocab_grade(label).pass_)

    def test_unknown_rejected(self):
        self.assertFalse(rec_ok("ship_it"))
        self.assertFalse(rec_ok(""))
        self.assertFalse(rec_vocab_grade("ok").pass_)

    def test_recs_set_frozen_size(self):
        self.assertEqual(RECS, {"high_confidence", "review_needed", "low_confidence"})

    def test_flops_proxy_scales(self):
        self.assertGreater(flops_proxy(16), 0.0)
        self.assertAlmostEqual(flops_proxy(16, bits=1.58), 16 * 1.58 / 16.0)


class CiGateWiresRecOk(unittest.TestCase):
    def test_gate_fails_on_bad_rec(self):
        g = grade_trajectory(["retrieve flagstaff", "assemble"], ["retrieve", "assemble"])
        self.assertTrue(ci_gate([g], recs=["high_confidence"]))
        self.assertFalse(ci_gate([g], recs=["ship_it"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
