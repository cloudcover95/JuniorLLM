"""Contracts for Enhanced TDA — works even when MLX/NumPy are absent."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reasoning.enhanced_tda import PersistencePair, TDASnapshot, EnhancedTDA, HAS_NUMPY, HAS_MLX


class PersistenceContract(unittest.TestCase):
    def test_persistence_non_negative(self):
        p = PersistencePair(birth=0.1, death=0.8, dimension=0)
        self.assertAlmostEqual(p.persistence, 0.7)
        q = PersistencePair(birth=0.9, death=0.2)
        self.assertEqual(q.persistence, 0.0)

    def test_snapshot_keys(self):
        s = TDASnapshot(
            bit_drift=0.1,
            spectral_norm=1.0,
            persistence_score=0.8,
            disagreement_score=0.2,
            n_pairs=3,
            top_persistence=[0.9],
            recommendation="high_confidence",
        )
        d = s.to_dict()
        self.assertEqual(d["recommendation"], "high_confidence")


class RecommendationVocab(unittest.TestCase):
    """Must stay aligned with JuniorBitNetFieldCore recommendations."""

    def test_shared_labels(self):
        shared = {"high_confidence", "review_needed", "low_confidence"}
        self.assertTrue(shared.issubset(shared))


@unittest.skipUnless(HAS_NUMPY or HAS_MLX, "numerical backend required for live analyze()")
class LiveAnalyze(unittest.TestCase):
    def test_analyze_returns_snapshot(self):
        eng = EnhancedTDA(dim=16, use_mlx=False)
        snap = eng.analyze([0.1, 0.2, 0.3, 0.0, 0.5] + [0.0] * 11)
        self.assertIn(snap.recommendation, {"high_confidence", "review_needed", "low_confidence"})
        self.assertTrue(0.0 <= snap.disagreement_score <= 1.0)
        self.assertIsInstance(eng.disagreement_gate(snap), bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
