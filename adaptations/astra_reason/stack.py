"""JuniorAstraReason entry — Fable gate + MemSys recall + rigid IQ + high-quant label."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from adaptations.astra_reason.config import AstraReasonConfig
from adaptations.astra_reason.memsys_bridge import Palace
from adaptations.astra_reason.rigid_iq import run_iq
from adaptations.fable.safety.classifier import FableStyleSafetyClassifier
from agent.guardrails import scan_prompt


@dataclass
class ReasonResult:
    ok: bool
    port: str
    family: str
    model_id: str
    quant: str
    loops: int
    rigidity: float
    recommendation: str
    memories: list[str]
    notes: str
    action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AstraReasonStack:
    def __init__(self, config: AstraReasonConfig | None = None, palace: Palace | None = None):
        self.config = config or AstraReasonConfig()
        self.palace = palace or Palace(max_slots=self.config.mem_slots)
        self.safety = FableStyleSafetyClassifier()

    def remember(self, key: str, value: str) -> None:
        self.palace.put(key, value)

    def reason(self, prompt: str, family: str | None = None) -> ReasonResult:
        cfg = self.config
        fam = family or cfg.family
        rail = scan_prompt(prompt)
        if not rail.ok:
            return ReasonResult(
                False, "JuniorAstraReason", fam, cfg.model_id(), cfg.quant,
                0, 0.0, "low_confidence", [], ";".join(rail.reasons), "refuse",
            )
        safety = self.safety.classify(prompt)
        if safety.action == "refuse":
            return ReasonResult(
                False, "JuniorAstraReason", fam, cfg.model_id(), cfg.quant,
                0, 0.0, "low_confidence", [], safety.reason, "refuse",
            )
        hits = self.palace.recall(prompt, k=3)
        mem_blob = " ".join(v for _, v, _ in hits)
        trace = run_iq(prompt + " " + mem_blob, loops=cfg.loops)
        notes = (
            f"stand-in only: {fam} {cfg.quant} + {cfg.loops} rigid BitNet loops; "
            f"mem={self.palace.backend} hits={len(hits)}"
        )
        if safety.action == "fallback":
            rec = "review_needed"
            action = "fallback"
        else:
            rec = trace.recommendation
            action = "allow"
        return ReasonResult(
            True,
            "JuniorAstraReason",
            fam,
            cfg.gemma_id if fam == "gemma" else cfg.qwen_id,
            cfg.quant,
            trace.loops,
            trace.rigidity,
            rec,
            [k for k, _, _ in hits],
            notes,
            action,
        )


_stack: AstraReasonStack | None = None


def get_astra_reason() -> AstraReasonStack:
    global _stack
    if _stack is None:
        _stack = AstraReasonStack()
    return _stack
