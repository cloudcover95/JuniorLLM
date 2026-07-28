# JuniorGemma-4 Portal

**Google Gemma 4 (Apache 2.0, April 2026) adapted for JuniorCloud BitNet edge stack**

## Why Gemma 4
- Fully open source under Apache 2.0 (commercially friendly)
- Sizes: Effective 2B, Effective 4B, 26B MoE, 31B Dense
- Built from Gemini 3 research
- High intelligence-per-parameter, designed for mobile / edge
- Strong reasoning + agentic workflows

## Adaptation Plan
- Primary targets: Gemma 4 4B or 26B-MoE quantized via existing BitNet 1.58 QuantConfig + BitNetLinear
- MLX backend first (Apple Silicon)
- Integrate SmartLayerOffloader + TDA manifold analysis
- Optional domain distillation for quant trading / SDK coding
- Fits the "no home lab" constraint immediately (small variants run on M4)

## Relationship to Other Portals
- Complements JuniorPortal-K3 (Kimi sparse MoE scale) and JuniorLLM-Fable (agentic reliability + safety)
- Best immediate practical edge model while Kimi distillation matures
- All three share BitNet core, production loops, vault security, rigid routing

## Status
Scaffolding and architecture committed. Next: BitNet quantization pipeline + MLX loader for Gemma 4 4B.
