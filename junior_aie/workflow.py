"""Durable workflow — checkpoint/resume on JuniorAstra Work (mini-Temporal)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Any

from adaptations.astra.work import WorkStore


class WorkflowEngine:
    def __init__(self, root: Path | None = None):
        self.store = WorkStore(root)

    def start(self, name: str, steps: list[str]) -> str:
        w = self.store.create(name, tokens_budget=2048)
        w.checkpoint = {"i": 0, "steps": steps, "results": []}
        w.status = "running"
        self.store.save(w)
        return w.work_id

    def resume(self, work_id: str, handler: Callable[[str], Any]) -> dict:
        w = self.store.load(work_id)
        if w is None:
            raise KeyError(work_id)
        ck = w.checkpoint
        i = int(ck.get("i", 0))
        steps = list(ck.get("steps", []))
        results = list(ck.get("results", []))
        while i < len(steps):
            results.append(handler(steps[i]))
            i += 1
            ck["i"] = i
            ck["results"] = results
            w.checkpoint = ck
            w.status = "paused" if i < len(steps) else "done"
            self.store.save(w)
        return {"work_id": work_id, "status": w.status, "results": results}
