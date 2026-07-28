"""
Fable-inspired Safety Classifier for JuniorLLM
==============================================
Transparent, offline-capable risk gate.
Routes high-risk requests to fallback / refuse paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RiskCategory(str, Enum):
    SAFE = "safe"
    CYBER = "cyber"
    BIO_CHEM = "bio_chem"
    DISTILLATION = "distillation"
    HIGH_STAKES_FINANCE = "high_stakes_finance"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    category: RiskCategory
    confidence: float
    reason: str
    action: str  # "allow" | "fallback" | "refuse"


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
    """Transparent safety gate used by IntentRouter."""

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


def classify_request(text: str) -> ClassificationResult:
    return FableStyleSafetyClassifier().classify(text)
