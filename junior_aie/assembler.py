"""Context Assembler — token-budgeted memory + retrieval + tools."""
from __future__ import annotations

from dataclasses import dataclass

from adaptations.astra.contextpipe import pack, PackedContext
from junior_aie.retrieval import RetrievalStack
from junior_aie.visibility import filter_hits, filter_memory


@dataclass
class Assembled:
    packed: PackedContext
    retrieved: list[str]
    tools: list[str]


class ContextAssembler:
    def __init__(self, retrieval: RetrievalStack | None = None, budget: int = 512):
        self.retrieval = retrieval or RetrievalStack()
        self.budget = budget

    def assemble(
        self,
        query: str,
        memory: list[tuple[str, str]],
        tools: list[str] | None = None,
        visibility: str = "public",
    ) -> Assembled:
        hits = filter_hits(self.retrieval.search(query, k=4), visibility)
        parts: list[tuple[str, str]] = [("query", query)]
        for key, val in filter_memory(memory, visibility):
            parts.append((f"mem:{key}", val))
        for doc, _score in hits:
            parts.append(("retr", doc))
        if tools:
            parts.append(("tools", ",".join(tools)))
        packed = pack(parts, budget=self.budget)
        return Assembled(packed=packed, retrieved=[d for d, _ in hits], tools=list(tools or []))
