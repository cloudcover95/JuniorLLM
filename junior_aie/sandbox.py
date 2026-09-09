"""Sandboxed tool executor — AST deny-list + bytecode, no subprocess."""
from __future__ import annotations

import ast
from typing import Any

from agent.guardrails import scan_skill_source

ALLOWED_BUILTINS = {"len": len, "range": range, "min": min, "max": max, "sum": sum, "abs": abs, "int": int, "float": float, "str": str, "list": list, "dict": dict}


class SandboxDenied(Exception):
    pass


def run_tool(src: str, args: dict[str, Any] | None = None, cpu_ops_hint: int = 10_000) -> Any:
    rail = scan_skill_source(src)
    if not rail.ok:
        raise SandboxDenied(";".join(rail.reasons))
    tree = ast.parse(src)
    # resource: reject huge trees
    if sum(1 for _ in ast.walk(tree)) > cpu_ops_hint:
        raise SandboxDenied("too_large")
    loc: dict[str, Any] = {"ARGS": args or {}}
    exec(compile(tree, "<sandbox>", "exec"), {"__builtins__": ALLOWED_BUILTINS}, loc)  # noqa: S102 — gated AST
    return loc.get("RESULT", None)
