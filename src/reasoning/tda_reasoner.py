"""
TDA Reasoner (Enhanced)
=======================
Localized LLM reasoning layer using the full Enhanced TDA engine:
Topological Data Analysis, Bit Drift, Feature Disagreement Scoring,
and SVD spectral topology — designed for BitNet edge embeddings.

Replaces the previous placeholder with the complete production implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.reasoning.enhanced_tda import EnhancedTDA, TDASnapshot, get_enhanced_tda

logger = logging.getLogger("juniorllm.tda_reasoner")

try:
    from bitnet_mlx.inference.ternary_analyzer import TernaryAnalyzer
    HAS_BITNET = True
except ImportError:
    HAS_BITNET = False
    TernaryAnalyzer = None


class TDAReasoner:
    """
    Production TDA Reasoner.

    Uses EnhancedTDA for structured, deterministic, low-latency decisions
    on top of optional BitNet ternary embeddings.
    """

    def __init__(self, output_dim: int = 128, drift_threshold: float = 0.85):
        self.tda = get_enhanced_tda(dim=output_dim, drift_threshold=drift_threshold)
        self.analyzer = TernaryAnalyzer(output_dim=output_dim) if HAS_BITNET else None
        logger.info("TDAReasoner (Enhanced) initialized | bitnet=%s", HAS_BITNET)

    def analyze_state(self, data: Any, ternary_hint: Optional[Any] = None) -> Dict[str, Any]:
        """
        Run full Enhanced TDA analysis.
        Returns a dict compatible with previous call sites plus new fields.
        """
        snapshot: TDASnapshot = self.tda.analyze(data, ternary_hint=ternary_hint)

        result = {
            "analysis": {
                "bit_drift": snapshot.bit_drift,
                "spectral_norm": snapshot.spectral_norm,
                "persistence": {
                    "persistence_score": snapshot.persistence_score,
                    "n_pairs": snapshot.n_pairs,
                    "top_persistence": snapshot.top_persistence,
                },
            },
            "disagreement_score": snapshot.disagreement_score,
            "recommendation": snapshot.recommendation,
            "gate_pass": self.tda.disagreement_gate(snapshot),
            "snapshot": snapshot.to_dict(),
        }

        if self.analyzer is not None:
            try:
                bitnet_analysis = self.analyzer.analyze(data)
                result["bitnet"] = bitnet_analysis
            except Exception as e:
                logger.debug("BitNet analyzer skipped: %s", e)

        return result

    def reason(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        TDA-enhanced reasoning entry point.

        If a numerical state (embedding / activation) is supplied, full
        Enhanced TDA is applied and used to gate confidence.
        """
        context = context or {}
        tda_result = None

        if state is not None:
            tda_result = self.analyze_state(state)
            context["tda"] = tda_result

        recommendation = (
            tda_result["recommendation"] if tda_result else "no_state_provided"
        )

        return {
            "prompt": prompt,
            "context": context,
            "reasoning": (
                f"Enhanced TDA reasoning complete. "
                f"Recommendation: {recommendation}. "
                f"Use disagreement_score / gate_pass for agentic confidence gates."
            ),
            "tda": tda_result,
            "recommendation": recommendation,
        }

    def gate(self, data: Any) -> bool:
        """Convenience: True if state is high-confidence."""
        snap = self.tda.analyze(data, update_baseline=False)
        return self.tda.disagreement_gate(snap)
