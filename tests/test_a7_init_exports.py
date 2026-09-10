"""A7 — live-beta package __init__ exports the 15-piece AIE surface."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import junior_aie


# One public name per numbered piece in junior_aie/README.md plus the facade.
PIECES = {
    "assembler": "ContextAssembler",
    "retrieval": "RetrievalStack",
    "router": "ModelRouter",
    "cache": "SemanticCache",
    "orchestrator": "Orchestrator",
    "mcp": "McpServer",
    "consensus": "decide",
    "sandbox": "run_tool",
    "guard_mw": "apply",
    "workflow": "WorkflowEngine",
    "stream": "stream_tokens",
    "tracer": "Tracer",
    "evalh": "grade_trajectory",
    "prompts": "PromptRegistry",
    "flywheel": "Flywheel",
    "framework": "build_framework",
}


class A7InitExportsTests(unittest.TestCase):
    def test_all_lists_fifteen_plus_facade(self):
        for name in PIECES.values():
            self.assertIn(name, junior_aie.__all__)
            self.assertTrue(hasattr(junior_aie, name), name)

    def test_fresh_clone_import(self):
        from junior_aie import ContextAssembler, ModelRouter, build_framework

        fw = build_framework()
        self.assertIsInstance(fw.assembler, ContextAssembler)
        self.assertIsInstance(fw.router, ModelRouter)
        route = fw.router.choose("astra-class reasoning", prefer="balanced")
        self.assertEqual(route.primary.name, "JuniorAstraReason")

    def test_no_eval_exec_in_package_init(self):
        src = (ROOT / "junior_aie" / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
