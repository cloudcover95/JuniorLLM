"""Facade — one object for the whole AI engineering stack."""
from __future__ import annotations

from junior_aie.assembler import ContextAssembler
from junior_aie.cache import SemanticCache
from junior_aie.consensus import Ballot, decide
from junior_aie.evalh import grade_trajectory, ci_gate
from junior_aie.flywheel import Flywheel
from junior_aie.guard_mw import apply as guard
from junior_aie.mcp import McpClient, McpServer
from junior_aie.orchestrator import Orchestrator
from junior_aie.prompts import PromptRegistry
from junior_aie.retrieval import RetrievalStack
from junior_aie.router import ModelRouter
from junior_aie.sandbox import run_tool
from junior_aie.stream import stream_tokens
from junior_aie.tracer import Tracer
from junior_aie.visibility import cache_key, normalize_visibility, scrub_public_text
from junior_aie.workflow import WorkflowEngine


class Framework:
    def __init__(self) -> None:
        self.retrieval = RetrievalStack()
        self.assembler = ContextAssembler(self.retrieval)
        self.router = ModelRouter()
        self.cache = SemanticCache()
        self.orch = Orchestrator()
        self.mcp = McpServer()
        self.prompts = PromptRegistry()
        self.flywheel = Flywheel()
        self.tracer = Tracer()

    def ask(
        self,
        query: str,
        memory: list[tuple[str, str]] | None = None,
        visibility: str = "public",
    ) -> dict:
        g = guard(query)
        if not g.ok:
            return {"ok": False, "reasons": g.reasons}
        vis = normalize_visibility(visibility)
        key = cache_key(g.text, vis)
        cached = self.cache.get(key)
        if cached:
            return {
                "ok": True,
                "cached": True,
                "text": cached,
                "visibility": vis,
                "hit_rate": self.cache.stats.hit_rate,
            }
        with self.tracer.span("assemble", q=g.text[:40]):
            assembled = self.assembler.assemble(
                g.text, memory or [], visibility=vis
            )
        route = self.router.choose(g.text)
        text = scrub_public_text(
            f"[{route.primary.name}] {assembled.packed.text[:200]}", vis
        )
        self.cache.put(key, text)
        return {
            "ok": True,
            "cached": False,
            "text": text,
            "visibility": vis,
            "port": route.primary.name,
            "fallbacks": route.fallbacks,
            "tokens": assembled.packed.tokens_est,
            "hit_rate": self.cache.stats.hit_rate,
        }


def build_framework() -> Framework:
    return Framework()
