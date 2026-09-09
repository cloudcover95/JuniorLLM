"""Streaming proxy — SSE frames + TTFT / ITL metrics."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator


@dataclass
class StreamStats:
    ttft_ms: float
    itl_ms: list[float]
    tokens: int

    def sse(self, text: str) -> str:
        return "".join(f"data: {part}\n\n" for part in text.split()) + "data: [DONE]\n\n"


def stream_tokens(parts: list[str]) -> tuple[Iterator[str], StreamStats]:
    t0 = time.perf_counter()
    itl: list[float] = []
    last = t0
    frames: list[str] = []
    ttft = 0.0
    for i, p in enumerate(parts):
        now = time.perf_counter()
        if i == 0:
            ttft = (now - t0) * 1000
        else:
            itl.append((now - last) * 1000)
        last = now
        frames.append(f"data: {p}\n\n")
    frames.append("data: [DONE]\n\n")
    stats = StreamStats(round(ttft, 4), [round(x, 4) for x in itl], len(parts))
    return iter(frames), stats
