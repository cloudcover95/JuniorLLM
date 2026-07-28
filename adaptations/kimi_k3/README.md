# JuniorPortal-K3 (Kimi K3 → BitNet Edge)

See the full architecture and scaffolding in the development workspace:

- Architecture: junior_bitnet/kimi_k3_adaptation/docs/ARCHITECTURE.md
- BitNetMoE + RigidRouter: core/bitnet_moe.py
- SmartExpertOffloader: moe/smart_expert_offloader.py
- Behavior card + production loop under the same tree

This pointer keeps JuniorLLM aware of the Kimi-derived edge MoE path without duplicating the large scaffolding here.

Target: practical edge install size far below the original 1.45 TB Kimi K3 weights while preserving long-context + sparse MoE ideas under 1.58-bit ternary + manifold-aware residency.
