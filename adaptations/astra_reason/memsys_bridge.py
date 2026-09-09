"""MemSys palace — local bit-drift slots compatible with JuniorMemSys-Suite.

If JuniorMemSys-Suite is on PYTHONPATH we try its API; otherwise we use
an in-tree dict palace so edge nodes stay offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from adaptations.astra_reason.rigid_iq import embed


def _try_external_put(key: str, value: str) -> bool:
    try:
        import junior_memsys  # type: ignore
    except Exception:
        return False
    fn = getattr(junior_memsys, "put", None) or getattr(junior_memsys, "store", None)
    if not callable(fn):
        return False
    fn(key, value)
    return True


@dataclass
class Palace:
    slots: dict[str, str] = field(default_factory=dict)
    vectors: dict[str, list[int]] = field(default_factory=dict)
    max_slots: int = 64
    backend: str = "in-tree"

    def put(self, key: str, value: str) -> None:
        if _try_external_put(key, value):
            self.backend = "JuniorMemSys-Suite"
        if len(self.slots) >= self.max_slots and key not in self.slots:
            oldest = next(iter(self.slots))
            self.slots.pop(oldest, None)
            self.vectors.pop(oldest, None)
        self.slots[key] = value
        self.vectors[key] = embed(key + " " + value)

    def get(self, key: str) -> str | None:
        return self.slots.get(key)

    def recall(self, query: str, k: int = 3) -> list[tuple[str, str, int]]:
        q = embed(query)
        scored: list[tuple[int, str]] = []
        for key, vec in self.vectors.items():
            score = sum(1 for a, b in zip(q, vec) if a == b and a != 0)
            scored.append((score, key))
        scored.sort(reverse=True)
        out = []
        for score, key in scored[:k]:
            out.append((key, self.slots[key], score))
        return out
