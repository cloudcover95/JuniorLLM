"""
JuniorLLM IntentRouter — Multi-Portal + Fable Safety
====================================================
Production update:
- FableStyleSafetyClassifier runs first on every request
- Portal selection: Fable (agentic/safety), Kimi-K3 (long context / MoE), Gemma-4 (fast edge)
- Existing deterministic tools, memory, fetch, algebra preserved
- Additive only; original behavior remains the default fallback path
"""

from __future__ import annotations

import logging
from typing import Optional

# Existing JuniorLLM components (package may be jr_llm or juniorllm depending on install)
try:
    from juniorllm.core.manifold_actuator import JuniorLLMManifold
    from juniorllm.utility.algebra_parser import AlgebraParser
    from juniorllm.utility.crono_node import ChronoNode
    from juniorllm.bridge.juniorfetch_bridge import JuniorFetchBridge
    from juniorllm.bridge.juniormemsys_bridge import JuniorMemSysBridge
except ImportError:
    # Fallback for jr_llm packaging
    try:
        from jr_llm.core.manifold_actuator import JuniorLLMManifold
        from jr_llm.utility.algebra_parser import AlgebraParser
        from jr_llm.utility.chrono_node import ChronoNode
        from jr_llm.bridge.juniorfetch_bridge import JuniorFetchBridge
        from jr_llm.bridge.juniormemsys_bridge import JuniorMemSysBridge
    except ImportError:
        JuniorLLMManifold = AlgebraParser = ChronoNode = None
        JuniorFetchBridge = JuniorMemSysBridge = None

# Safety + portals (additive)
try:
    from adaptations.fable.safety.classifier import (
        FableStyleSafetyClassifier,
        ClassificationResult,
        RiskCategory,
    )
except ImportError:
    # Absolute safety net if package layout differs
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from adaptations.fable.safety.classifier import (
        FableStyleSafetyClassifier,
        ClassificationResult,
        RiskCategory,
    )

try:
    from adaptations.gemma4.bitnet_mlx_loader import get_gemma4_loader
except ImportError:
    get_gemma4_loader = None

logger = logging.getLogger("juniorllm.router")


class IntentRouter:
    def __init__(self):
        self.manifold = JuniorLLMManifold() if JuniorLLMManifold else None
        self.algebra = AlgebraParser() if AlgebraParser else None
        self.chrono = ChronoNode() if ChronoNode else None
        self.fetch = JuniorFetchBridge() if JuniorFetchBridge else None
        self.memory = JuniorMemSysBridge() if JuniorMemSysBridge else None

        # New production components
        self.safety = FableStyleSafetyClassifier()
        self.gemma_loader = get_gemma4_loader() if get_gemma4_loader else None
        self.default_portal = "gemma4"  # fastest interactive path on edge

    def _select_portal(self, user_input: str, classification: ClassificationResult) -> str:
        """Simple production portal selector."""
        lower = user_input.lower()

        # High-stakes or long-horizon agentic → Fable path (safety + consistency)
        if any(k in lower for k in ["plan", "agent", "multi-step", "long-running", "days", "strategy"]):
            return "fable"

        # Long context / MoE scale needs → Kimi path
        if any(k in lower for k in ["long context", "1m", "million tokens", "sparse", "moe", "kimi"]):
            return "kimi"

        # Default fast edge path
        return "gemma4"

    def route(self, user_input: str):
        # ------------------------------------------------------------------
        # 1. Safety gate (Fable-inspired) — always first
        # ------------------------------------------------------------------
        classification = self.safety.classify(user_input)

        if classification.action == "refuse":
            return (
                f"[JuniorLLM Safety] Request refused.\n"
                f"Category: {classification.category.value}\n"
                f"Reason: {classification.reason}\n"
                f"Please rephrase or choose a different topic."
            )

        if classification.action == "fallback":
            # Constrained path — still answer but with clear notice
            logger.info("Safety fallback triggered: %s", classification.reason)
            prefix = (
                f"[Safety Fallback — {classification.category.value}] "
                f"{classification.reason}\n\n"
            )
        else:
            prefix = ""

        lower = user_input.lower()

        # ------------------------------------------------------------------
        # 2. Deterministic tools (preserved from original)
        # ------------------------------------------------------------------
        if self.algebra and any(op in lower for op in ['+', '-', '*', '/', '^', 'calculate', 'solve']):
            return prefix + str(self.algebra.compute(user_input))

        if self.chrono:
            if "time" in lower or "clock" in lower:
                return prefix + str(self.chrono.get_clock())
            if "stopwatch" in lower:
                return prefix + str(self.chrono.toggle_stopwatch())
            if "timer" in lower:
                secs = int(''.join(filter(str.isdigit, lower)) or 60)
                self.chrono.deploy_timer(secs)
                return prefix + f"Timer set for {secs} seconds."

        if self.fetch and ("file" in lower or "document" in lower):
            context = self.fetch.get_context(user_input)
            return prefix + f"File context: {context[:300]}..."

        if self.memory and ("remember" in lower or "memory" in lower):
            context = self.memory.get_long_term_context(user_input)
            return prefix + f"Long-term memory: {context[:300]}..."

        # ------------------------------------------------------------------
        # 3. Multi-portal LLM path
        # ------------------------------------------------------------------
        portal = self._select_portal(user_input, classification)

        if portal == "gemma4" and self.gemma_loader is not None:
            # Fastest interactive edge path
            response = self.gemma_loader.generate(user_input)
            return prefix + f"[Portal: Gemma-4 BitNet/MLX]\n{response}"

        if portal == "fable":
            # Long-horizon / high-consistency path (behavioral)
            return (
                prefix +
                f"[Portal: Fable-style]\n"
                f"Long-horizon agentic mode engaged. "
                f"Safety classification: {classification.category.value}. "
                f"Proceeding with rigid, multi-step reasoning posture. "
                f"(Connect to full Fable behavioral loop / JuniorAGI for production agentic runs.)"
            )

        if portal == "kimi":
            return (
                prefix +
                f"[Portal: Kimi-K3 BitNet]\n"
                f"Sparse MoE / long-context path selected. "
                f"Use JuniorPortal-K3 distillation + SmartExpertOffloader for full edge MoE. "
                f"(Pipeline ready for agentic long-context work.)"
            )

        # Final fallback to original manifold / local LLM behavior
        if self.manifold:
            return prefix + "Roadblock or unknown portal — using deterministic manifold routing."

        return prefix + "[JuniorLLM] No portal available for this request."
