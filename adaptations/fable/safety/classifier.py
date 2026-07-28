"""
Fable-inspired Safety Classifier for JuniorLLM

Lightweight, transparent classifier that routes high-risk requests
to a constrained path or refusal, mirroring the spirit of Claude Fable 5's
external safeguards while remaining fully local and BitNet-native.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class RiskCategory(str, Enum):
    SAFE = "safe"
    CYBER = "cyber"
    BIO_CHEM = "bio_chem"
    DISTILLATION = "distillation"
    HIGH_STAKES_FINANCE = "high_stakes_finance"  # optional stricter mode
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    category: RiskCategory
    confidence: float
    reason: str
    action: str  # "allow" | "fallback" | "refuse"


# Simple keyword / pattern heuristics for scaffolding.
# Production version should be a small BitNet classifier or hybrid.
CYBER_PATTERNS = [
    "exploit", "payload", "ransomware", "zero-day", "c2 server",
    "lateral movement", "privilege escalation", "malware",
]
BIO_PATTERNS = [
    "synthesize pathogen", "gain of function", "select agent",
    "bioweapon", "toxin production",
]
DISTILL_PATTERNS = [
    "distill the model", "extract weights", "replicate the system prompt",
    "copy the full architecture", "train a student on your outputs",
]


class FableStyleSafetyClassifier:
    """
    Transparent safety gate.
    Returns a ClassificationResult that the runtime uses to decide
    whether to run the full JuniorLLM-Fable path, a constrained path, or refuse.
    """

    def classify(self, user_text: str) -> ClassificationResult:
        text = user_text.lower()

        for p in CYBER_PATTERNS:
            if p in text:
                return ClassificationResult(
                    category=RiskCategory.CYBER,
                    confidence=0.85,
                    reason=f"Matched cyber-related pattern: '{p}'",
                    action="fallback",
                )
        for p in BIO_PATTERNS:
            if p in text:
                return ClassificationResult(
                    category=RiskCategory.BIO_CHEM,
                    confidence=0.9,
                    reason=f"Matched bio/chem pattern: '{p}'",
                    action="refuse",
                )
        for p in DISTILL_PATTERNS:
            if p in text:
                return ClassificationResult(
                    category=RiskCategory.DISTILLATION,
                    confidence=0.8,
                    reason=f"Matched distillation pattern: '{p}'",
                    action="refuse",
                )

        return ClassificationResult(
            category=RiskCategory.SAFE,
            confidence=0.7,
            reason="No high-risk patterns detected",
            action="allow",
        )


# Convenience
def classify_request(text: str) -> ClassificationResult:
    return FableStyleSafetyClassifier().classify(text)
