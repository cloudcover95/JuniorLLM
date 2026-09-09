"""Prompt registry — versioning + A/B + rollback."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromptVer:
    name: str
    version: int
    body: str
    active: bool = False


class PromptRegistry:
    def __init__(self) -> None:
        self.items: dict[str, list[PromptVer]] = {}

    def publish(self, name: str, body: str) -> PromptVer:
        hist = self.items.setdefault(name, [])
        ver = PromptVer(name, len(hist) + 1, body, active=True)
        for p in hist:
            p.active = False
        hist.append(ver)
        return ver

    def rollback(self, name: str, version: int) -> PromptVer:
        hist = self.items[name]
        for p in hist:
            p.active = p.version == version
        return next(p for p in hist if p.active)

    def ab(self, name: str, bucket: int) -> PromptVer:
        hist = self.items[name]
        if len(hist) == 1:
            return hist[0]
        return hist[-1] if bucket % 2 else hist[-2]
