"""JuniorAstra Edge Runner — execute Work on local BitNet + Fable gate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptations.astra.contextpipe import pack
from adaptations.astra.work import Work, WorkStore
from adaptations.fable.safety.classifier import FableStyleSafetyClassifier
from agent.guardrails import scan_prompt
from ports.registry import pick


@dataclass
class RunResult:
    work_id: str
    status: str
    action: str
    port: str
    notes: str
    context_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "status": self.status,
            "action": self.action,
            "port": self.port,
            "notes": self.notes,
            "context_tokens": self.context_tokens,
        }


class AstraRunner:
    def __init__(self, store: WorkStore | None = None):
        self.store = store or WorkStore()
        self.safety = FableStyleSafetyClassifier()

    def start(self, goal: str, extra_notes: str = "", tokens_budget: int = 1024) -> RunResult:
        work = self.store.create(goal, tokens_budget=tokens_budget)
        return self.step(work.work_id, extra_notes=extra_notes)

    def step(self, work_id: str, extra_notes: str = "") -> RunResult:
        work = self.store.load(work_id)
        if work is None:
            return RunResult(work_id, "blocked", "refuse", "none", "work_not_found", 0)

        rail = scan_prompt(work.goal + " " + extra_notes)
        if not rail.ok:
            work.status = "blocked"
            work.events.append({"kind": "guardrail", "reasons": rail.reasons})
            self.store.save(work)
            return RunResult(work.work_id, "blocked", "refuse", "none", ";".join(rail.reasons), 0)

        safety = self.safety.classify(work.goal)
        if safety.action == "refuse":
            work.status = "blocked"
            work.events.append({"kind": "fable_refuse", "reason": safety.reason})
            self.store.save(work)
            return RunResult(work.work_id, "blocked", "refuse", "JuniorFable", safety.reason, 0)

        packed = pack(
            [
                ("goal", work.goal),
                ("checkpoint", str(work.checkpoint)),
                ("notes", extra_notes),
            ],
            budget=min(512, work.tokens_budget - work.tokens_used),
        )
        work.tokens_used += packed.tokens_est
        port = pick(work.goal, ram_gb=8.0)
        work.status = "running" if safety.action == "allow" else "paused"
        work.checkpoint = {"last_pack": packed.provenance, "port": port.name}
        work.events.append({"kind": "step", "port": port.name, "tokens": packed.tokens_est})
        if work.tokens_used >= work.tokens_budget:
            work.status = "paused"
            work.events.append({"kind": "budget_pause"})
        self.store.save(work)
        return RunResult(
            work.work_id,
            work.status,
            safety.action,
            port.name,
            packed.text[:240],
            packed.tokens_est,
        )

    def resume(self, work_id: str) -> RunResult:
        return self.step(work_id, extra_notes="resume from checkpoint")
