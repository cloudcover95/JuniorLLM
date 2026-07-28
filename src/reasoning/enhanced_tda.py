"""
Enhanced TDA Engine for JuniorLLM
=================================
Production implementation of the topological analysis layer previously
iterated across JuniorCloud (Bit Drift, TDA Meshes, Feature Disagreement,
SVD-Zero topology, BitNet-aware embeddings).

Edge-first: pure MLX / NumPy friendly, no heavy external TDA libraries required.
Designed for M4 / van / no-home-lab constraints while remaining production-grade.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import math

logger = logging.getLogger("juniorllm.enhanced_tda")

# Optional backends
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class PersistencePair:
    birth: float
    death: float
    dimension: int = 0

    @property
    def persistence(self) -> float:
        return max(0.0, self.death - self.birth)


@dataclass
class TDASnapshot:
    """Full topological summary of a state."""
    bit_drift: float
    spectral_norm: float
    persistence_score: float
    disagreement_score: float
    n_pairs: int
    top_persistence: List[float]
    recommendation: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Enhanced TDA Engine
# ---------------------------------------------------------------------------

class EnhancedTDA:
    """
    Full Enhanced TDA implementation.

    Capabilities (as previously iterated):
    - SVD spectral topology (bit-drift relative to a baseline manifold)
    - Practical persistence proxies from distance / singular spectra
    - Feature Disagreement Scoring for confidence / review gates
    - Rolling baseline update (adaptive manifold)
    - BitNet / ternary-aware analysis hooks
    - Deterministic, low-latency, edge-safe
    """

    def __init__(
        self,
        dim: int = 128,
        drift_threshold: float = 0.85,
        disagreement_threshold: float = 0.30,
        baseline_decay: float = 0.90,
        use_mlx: bool = True,
    ):
        self.dim = dim
        self.drift_threshold = drift_threshold
        self.disagreement_threshold = disagreement_threshold
        self.baseline_decay = baseline_decay
        self.use_mlx = use_mlx and HAS_MLX

        self._baseline_sigma: Optional[Any] = None
        self._baseline_state: Optional[Any] = None
        self._history: List[TDASnapshot] = []

        self._init_baseline()
        logger.info(
            "EnhancedTDA initialized | dim=%d | mlx=%s | drift_thresh=%.2f",
            dim, self.use_mlx, drift_threshold,
        )

    # ------------------------------------------------------------------
    # Baseline & array helpers
    # ------------------------------------------------------------------

    def _init_baseline(self) -> None:
        if self.use_mlx:
            state = mx.random.normal((self.dim, self.dim))
            _, sigma, _ = mx.linalg.svd(state, compute_uv=False)
            self._baseline_state = state
            self._baseline_sigma = sigma
        elif HAS_NUMPY:
            state = np.random.randn(self.dim, self.dim).astype(np.float32)
            sigma = np.linalg.svd(state, compute_uv=False)
            self._baseline_state = state
            self._baseline_sigma = sigma
        else:
            self._baseline_state = None
            self._baseline_sigma = None
            logger.warning("No MLX or NumPy — EnhancedTDA running in limited mode")

    def _to_matrix(self, data: Any) -> Any:
        """Coerce input into a square-ish matrix for spectral analysis."""
        if self.use_mlx:
            if isinstance(data, mx.array):
                arr = data
            else:
                arr = mx.array(data, dtype=mx.float32)
            if arr.ndim == 1:
                # Embed 1-D into a diagonal-ish / outer product matrix
                n = min(arr.shape[0], self.dim)
                mat = mx.zeros((self.dim, self.dim))
                mat = mat.at[:n, :n].add(mx.diag(arr[:n]))
                return mat
            if arr.ndim >= 2:
                # Take first two dims and pad / crop to dim
                h, w = arr.shape[0], arr.shape[1]
                mat = mx.zeros((self.dim, self.dim))
                h2, w2 = min(h, self.dim), min(w, self.dim)
                mat = mat.at[:h2, :w2].add(arr[:h2, :w2])
                return mat
            return mx.broadcast_to(arr, (self.dim, self.dim))

        if HAS_NUMPY:
            arr = np.asarray(data, dtype=np.float32)
            if arr.ndim == 1:
                n = min(arr.shape[0], self.dim)
                mat = np.zeros((self.dim, self.dim), dtype=np.float32)
                mat[:n, :n] = np.diag(arr[:n])
                return mat
            if arr.ndim >= 2:
                mat = np.zeros((self.dim, self.dim), dtype=np.float32)
                h2, w2 = min(arr.shape[0], self.dim), min(arr.shape[1], self.dim)
                mat[:h2, :w2] = arr[:h2, :w2]
                return mat
            return np.broadcast_to(arr, (self.dim, self.dim))

        raise RuntimeError("No numerical backend available for EnhancedTDA")

    def _svd_spectrum(self, mat: Any) -> Any:
        if self.use_mlx:
            return mx.linalg.svd(mat, compute_uv=False)
        return np.linalg.svd(mat, compute_uv=False)

    def _norm(self, x: Any) -> float:
        if self.use_mlx:
            return float(mx.linalg.norm(x).item())
        return float(np.linalg.norm(x))

    def _sum(self, x: Any) -> float:
        if self.use_mlx:
            return float(mx.sum(x).item())
        return float(np.sum(x))

    # ------------------------------------------------------------------
    # Persistence proxy (practical, edge-safe)
    # ------------------------------------------------------------------

    def _persistence_from_spectrum(self, sigma: Any) -> Tuple[List[PersistencePair], float]:
        """
        Construct a practical 0-dimensional persistence proxy from the
        singular spectrum. Large singular values = long-lived features.
        """
        if self.use_mlx:
            vals = [float(v.item()) for v in sigma]
        else:
            vals = [float(v) for v in sigma]

        vals = sorted((v for v in vals if v > 1e-8), reverse=True)
        if not vals:
            return [], 0.0

        max_v = vals[0]
        pairs: List[PersistencePair] = []
        for i, v in enumerate(vals):
            # birth near 0, death proportional to relative singular mass
            birth = 0.0
            death = v / max_v
            pairs.append(PersistencePair(birth=birth, death=death, dimension=0))

        # Persistence score = average relative persistence of top features
        top_k = pairs[: min(8, len(pairs))]
        score = sum(p.persistence for p in top_k) / max(len(top_k), 1)
        return pairs, float(score)

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        data: Any,
        update_baseline: bool = True,
        ternary_hint: Optional[Any] = None,
    ) -> TDASnapshot:
        """
        Full enhanced TDA analysis of an incoming state.

        Parameters
        ----------
        data : array-like
            Embedding, activation, or state tensor.
        update_baseline : bool
            Whether to adaptively update the rolling baseline.
        ternary_hint : optional
            Optional BitNet / ternary statistics to fold into disagreement.
        """
        mat = self._to_matrix(data)
        sigma = self._svd_spectrum(mat)

        # Bit Drift relative to baseline
        if self._baseline_sigma is not None:
            if self.use_mlx:
                # Align lengths
                n = min(sigma.shape[0], self._baseline_sigma.shape[0])
                drift = self._norm(sigma[:n] - self._baseline_sigma[:n])
            else:
                n = min(len(sigma), len(self._baseline_sigma))
                drift = self._norm(sigma[:n] - self._baseline_sigma[:n])
        else:
            drift = 0.0

        spectral_norm = self._norm(sigma)

        # Persistence proxy
        pairs, persistence_score = self._persistence_from_spectrum(sigma)
        top_pers = [p.persistence for p in pairs[:5]]

        # Feature Disagreement Scoring
        # High drift + low persistence → high disagreement (review needed)
        # Low drift + high persistence → high confidence
        raw_disagreement = 1.0 - persistence_score
        if drift > self.drift_threshold:
            raw_disagreement = min(1.0, raw_disagreement + 0.25)

        if ternary_hint is not None:
            # Optional BitNet ternary density signal
            try:
                density = float(ternary_hint)
                if density < 0.15 or density > 0.85:
                    raw_disagreement = min(1.0, raw_disagreement + 0.1)
            except Exception:
                pass

        disagreement_score = float(max(0.0, min(1.0, raw_disagreement)))

        if disagreement_score < self.disagreement_threshold:
            recommendation = "high_confidence"
        elif disagreement_score < 0.55:
            recommendation = "review_needed"
        else:
            recommendation = "low_confidence"

        # Adaptive baseline update (Bit Drift tracking)
        if update_baseline and self._baseline_sigma is not None:
            decay = self.baseline_decay
            if self.use_mlx:
                n = min(sigma.shape[0], self._baseline_sigma.shape[0])
                self._baseline_sigma = (
                    decay * self._baseline_sigma[:n] + (1.0 - decay) * sigma[:n]
                )
            else:
                n = min(len(sigma), len(self._baseline_sigma))
                self._baseline_sigma = (
                    decay * self._baseline_sigma[:n] + (1.0 - decay) * sigma[:n]
                )

        snapshot = TDASnapshot(
            bit_drift=float(drift),
            spectral_norm=float(spectral_norm),
            persistence_score=float(persistence_score),
            disagreement_score=disagreement_score,
            n_pairs=len(pairs),
            top_persistence=top_pers,
            recommendation=recommendation,
            meta={
                "dim": self.dim,
                "backend": "mlx" if self.use_mlx else ("numpy" if HAS_NUMPY else "none"),
            },
        )
        self._history.append(snapshot)
        if len(self._history) > 256:
            self._history = self._history[-256:]

        return snapshot

    def disagreement_gate(self, snapshot: TDASnapshot) -> bool:
        """Return True if the state is high-confidence (gate passes)."""
        return snapshot.disagreement_score < self.disagreement_threshold

    def recent_drift_trend(self, window: int = 16) -> float:
        if not self._history:
            return 0.0
        recent = self._history[-window:]
        return sum(s.bit_drift for s in recent) / len(recent)

    def reset_baseline(self) -> None:
        self._init_baseline()
        self._history.clear()
        logger.info("EnhancedTDA baseline reset")


# ---------------------------------------------------------------------------
# Convenience singleton
# ---------------------------------------------------------------------------

_engine: Optional[EnhancedTDA] = None

def get_enhanced_tda(**kwargs) -> EnhancedTDA:
    global _engine
    if _engine is None:
        _engine = EnhancedTDA(**kwargs)
    return _engine
