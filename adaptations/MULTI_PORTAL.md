# JuniorLLM Multi-Portal Status (2026-09-09)

Four complementary portals under one BitNet edge-native substrate.
Built for real edge hardware (M4, van solar builds). No home lab assumed.

## Portals

1. **JuniorPortal-K3** (Kimi K3 open weights)
   - BitNet 1.58 re-architecture + expert pruning + SmartExpertOffloader
   - Pointer: adaptations/kimi_k3/

2. **JuniorLLM-Fable** (Claude Fable 5 style)
   - Behavioral + long-horizon agentic + transparent safety classifiers
   - Files: adaptations/fable/

3. **JuniorGemma-4** (Google Gemma 4 Apache 2.0)
   - High IQ/param, mobile-first, BitNet + MLX
   - Files: adaptations/gemma4/

4. **JuniorAstra** (open-source Astra runtime capabilities)
   - Learns from Apache-2.0 matrixorigin/Astra: durable Work, ContextPipe, User Runner, checkpoints
   - **Not** GPT-6 Astra (closed weights / no local download)
   - Files: adaptations/astra/
   - BYO LLM: FieldCore / Gemma4 / BitNet-2B4T

## Shared Infrastructure
- BitNet 1.58 ternary core
- SVD-Zero + TDA + NightTernary
- Fable safety first, then portal select

## Policy
All commits are additive. No existing files or repositories are deleted.
