# JuniorLLM Adaptations

This directory holds **additive** adaptations of frontier models into the JuniorCloud BitNet edge-native stack.

## Current Portals

### 1. JuniorLLM-Fable (Claude Fable 5 style)
- Path: `adaptations/fable/`
- Focus: Long-horizon agentic reliability, high consistency, Fable-inspired safety classifiers + transparent fallbacks
- Status: Architecture + safety classifier + behavior card committed

### 2. JuniorPortal-K3 (Kimi K3 BitNet)
- Full scaffolding lives under the broader `junior_bitnet/kimi_k3_adaptation/` workspace and is mirrored conceptually here
- Focus: Sparse MoE scale, 1.58-bit experts, practical edge install size (target 30–80 GB vs original ~1.45 TB), rigid routing, SmartExpertOffloader
- Status: Complete architecture + BitNetMoE + offloader + adaptation loop

Both paths share the same BitNet core, TDA/SVD-Zero, production loops, and vault security posture. They are designed to coexist; a future model router can select or blend by task type.

**Policy**: Never delete or overwrite existing files in this repo or sibling JuniorCloudLLC repositories when adding adaptations.
