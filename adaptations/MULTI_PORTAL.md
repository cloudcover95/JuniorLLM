# JuniorLLM Multi-Portal Status (2026-07-28)

Three complementary portals under one BitNet edge-native substrate.
Built for real edge hardware (M4, van solar builds). No home lab assumed.

## Portals

1. **JuniorPortal-K3** (Kimi K3 open weights)
   - BitNet 1.58 re-architecture + expert pruning + SmartExpertOffloader
   - Target practical install 30-80 GB vs original ~1.45 TB
   - Pointer: adaptations/kimi_k3/

2. **JuniorLLM-Fable** (Claude Fable 5 style)
   - Behavioral + long-horizon agentic + transparent safety classifiers
   - No weight download required
   - Files: adaptations/fable/

3. **JuniorGemma-4** (Google Gemma 4 Apache 2.0)
   - High IQ/param, mobile-first, 2B/4B/26B sizes ideal for edge
   - BitNet quantization + MLX native path
   - Files: adaptations/gemma4/

## Shared Infrastructure
- BitNet 1.58 ternary core
- SVD-Zero + TDA manifold analysis
- Smart offloading
- Production loops + CI modernization
- Vault security / risk templates
- Rigid routing for high-stakes paths

## Router
Fable-style SafetyClassifier runs first. Then select or blend portal by task type (coding / quant / general agentic).

## Policy
All commits are additive. No existing files or repositories are deleted or overwritten.

Path: architecture complete → runnable skeletons → domain distillation + eval → live beta on M4.
