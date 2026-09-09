# Junior AI Engineering Framework

Fifteen local pieces. No LangChain. No MCP SDK. BitNet ports via `ports.registry`.

| # | Piece | Module |
|---|--------|--------|
| 1 | Context Assembler | `assembler.py` |
| 2 | Retrieval (chunk + BM25 + dense + rerank) | `retrieval.py` |
| 3 | Model Router | `router.py` |
| 4 | Semantic Cache | `cache.py` |
| 5 | Agent Orchestrator | `orchestrator.py` |
| 6 | MCP JSON-RPC | `mcp.py` |
| 7 | Multi-agent consensus | `consensus.py` |
| 8 | Sandboxed executor | `sandbox.py` |
| 9 | Guardrails MW | `guard_mw.py` |
| 10 | Durable workflow | `workflow.py` |
| 11 | Streaming proxy | `stream.py` |
| 12 | Tracer | `tracer.py` |
| 13 | Eval harness | `evalh.py` |
| 14 | Prompt registry | `prompts.py` |
| 15 | Data flywheel | `flywheel.py` |

```python
from junior_aie import build_framework
fw = build_framework()
fw.retrieval.add("Flagstaff granite is often dry in late summer")
print(fw.ask("flagstaff conditions", memory=[("covenant", "no private land without consent")]))
```
