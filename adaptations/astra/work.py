"""JuniorAstra durable Work — open Astra capability, local store."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Work:
    work_id: str
    goal: str
    status: str = "open"  # open | running | paused | blocked | done
    tokens_budget: int = 4096
    tokens_used: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorkStore:
    def __init__(self, root: Path | None = None):
        self.root = root or Path("data/astra/works")
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, work_id: str) -> Path:
        return self.root / f"{work_id}.json"

    def create(self, goal: str, tokens_budget: int = 4096) -> Work:
        w = Work(
            work_id=uuid.uuid4().hex[:12],
            goal=goal,
            tokens_budget=tokens_budget,
            created_at=_now(),
            updated_at=_now(),
        )
        w.events.append({"at": w.created_at, "kind": "created", "goal": goal})
        self.save(w)
        return w

    def save(self, work: Work) -> None:
        work.updated_at = _now()
        self._path(work.work_id).write_text(json.dumps(work.to_dict(), indent=2), encoding="utf-8")

    def load(self, work_id: str) -> Work | None:
        p = self._path(work_id)
        if not p.exists():
            return None
        raw = json.loads(p.read_text(encoding="utf-8"))
        return Work(**raw)
