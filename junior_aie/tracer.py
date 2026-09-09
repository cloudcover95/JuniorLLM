"""LLM tracer — OpenTelemetry-style spans as JSON (no OTLP vendor)."""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start: float
    end: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"


class Tracer:
    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex[:16]
        self.spans: list[Span] = []
        self._stack: list[str] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        sid = uuid.uuid4().hex[:8]
        parent = self._stack[-1] if self._stack else None
        sp = Span(name, self.trace_id, sid, parent, time.perf_counter(), attrs=dict(attrs))
        self._stack.append(sid)
        self.spans.append(sp)
        try:
            yield sp
        except Exception as exc:
            sp.status = "error"
            sp.attrs["error"] = str(exc)
            raise
        finally:
            sp.end = time.perf_counter()
            self._stack.pop()

    def export(self) -> list[dict]:
        return [asdict(s) for s in self.spans]
