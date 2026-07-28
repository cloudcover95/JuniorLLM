# Agentic Pipelines Across All Three Portals

Production plan for long-horizon / agentic work inside JuniorLLM.

## Shared Agentic Substrate
- IntentRouter (now multi-portal + Fable safety)
- Production loops (multi_portal_production_loop.py)
- JuniorAGI / TradingAgents state machines (existing ecosystem)
- Tool allow-lists + sandbox (vault + existing bridges)

## Portal-Specific Agentic Roles

### Gemma-4 BitNet/MLX (fast edge)
- Interactive coding assistant
- Short-to-medium agentic loops (tool use, file context, memory)
- Primary interactive model for day-to-day SDK and quant work on M4

### Fable-style path
- Long-horizon multi-day plans
- High-consistency strategy / research / complex coding projects
- Safety-first posture for any elevated risk
- Integrates with JuniorAGI hierarchical + event-driven automata

### Kimi-K3 BitNet path
- Extremely long context agentic sessions
- Sparse MoE expert specialization (once distilled)
- Heavy retrieval + multi-document reasoning on edge via SmartExpertOffloader

## Agentic Work Pipeline (Scripted)

1. **Safety gate** (Fable classifier) — every request
2. **Portal selection** — IntentRouter chooses Gemma / Fable / Kimi
3. **State machine** — existing JuniorAGI / TradingAgents loops take over for multi-step goals
4. **Tool execution** — JuniorFetch, JuniorMemSys, algebra, chrono, future SDK tools
5. **Verifier / red-team** — continuous eval harness + vault risk templates
6. **Persistence** — multi_portal_state.json + existing audit ledger

## Next Concrete Scripts
- Wire Gemma4BitNetLoader.generate into a real MLX forward once quantized checkpoint exists
- Add domain distillation data loaders from JuniorStock / SDK traces
- Extend multi_portal_production_loop with latency / memory / accuracy gates
- Optional: explicit AgenticPlanner class that issues multi-step plans under Fable path

All of the above remains additive and edge-first.
