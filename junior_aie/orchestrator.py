"""Agent orchestrator — deterministic state machine. No LangChain."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

STATES = ("INIT", "PLAN", "ACT", "CHECK", "DONE", "FAIL")


@dataclass
class Machine:
    state: str = "INIT"
    log: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def advance(self, nxt: str, note: str = "") -> None:
        if nxt not in STATES:
            raise ValueError(nxt)
        self.log.append(f"{self.state}->{nxt}:{note}")
        self.state = nxt


class Orchestrator:
    def run(self, plan_fn: Callable[[], str], act_fn: Callable[[str], str], check_fn: Callable[[str], bool]) -> Machine:
        m = Machine()
        m.advance("PLAN", "start")
        plan = plan_fn()
        m.payload["plan"] = plan
        m.advance("ACT", plan[:80])
        out = act_fn(plan)
        m.payload["out"] = out
        m.advance("CHECK")
        ok = check_fn(out)
        m.advance("DONE" if ok else "FAIL", "pass" if ok else "refute")
        return m
