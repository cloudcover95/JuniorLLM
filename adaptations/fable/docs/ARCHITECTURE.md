# JuniorLLM-Fable / JuniorPortal-Fable Architecture

**Behavioral + Architectural Adaptation of Anthropic Claude Fable 5 strengths into BitNet Edge-Native JuniorLLM**

**Important**: Claude Fable 5 (Anthropic, June 2026) is a proprietary Mythos-class model with **no public weights**. This is therefore a **distillation-style / behavioral / architectural adaptation**, not a weight conversion. We capture the qualities that made Fable 5 exceptional (long-horizon agentic work, strong coding, finance/reasoning, high consistency, sophisticated safety classifiers) and implement them natively inside the JuniorCloud BitNet + TDA + rigid-loop ecosystem.

**Do not remove or overwrite any existing Kimi-K3 adaptation or other repos.** This lives alongside JuniorPortal-K3.

---

## 1. What We Are Capturing from Fable 5

From public descriptions and system card themes:
- Mythos-class long-horizon agentic capability (can run for days on complex plans)
- Exceptional software engineering + knowledge work + scientific / analytical reasoning
- Strong finance / complex judgment performance
- 1M context, high output length
- Sophisticated external safety classifiers that fall back to a weaker model on high-risk domains (cyber, bio, chemistry, distillation)
- High consistency and fewer “blunders” on complex multi-step plans (community preference over later Opus variants in some workflows)

## 2. JuniorCloud Translation

| Fable 5 Strength | JuniorLLM-Fable Implementation |
|------------------|--------------------------------|
| Long-horizon agentic | Production loops + Hierarchical + Event-Driven state machines already in TradingAgents / JuniorAGI |
| High consistency / low blunder | Rigid routing mode + stronger shared experts + TDA regularization (same rigidity philosophy as Kimi adaptation) |
| Safety classifiers + fallback | Explicit SafetyClassifier layer that routes high-risk requests to a smaller / more constrained BitNet path or refuses |
| Finance / quant strength | Domain distillation targets from JuniorStock + TradingAgents traces |
| Coding excellence | Tight integration with JuniorSDK loops and code-generation verifiers |
| Edge feasibility | Full BitNet 1.58 + SmartExpertOffloader + MLX; target practical install size |

## 3. Architecture Overview

```
JuniorLLM-Fable Runtime
│
├── Portal Orchestrator (shared with Kimi path where possible)
│   ├── Rigid / Deterministic routing for high-stakes
│   ├── SafetyClassifier (Fable-inspired)
│   └── Manifold-aware expert / layer selection (TDA + SVD-Zero)
│
├── Core BitNet Stack (reuses junior_bitnet)
│   ├── BitNetMoE (shared with Kimi adaptation where sensible)
│   ├── Shared Experts (high rigidity)
│   └── SmartExpertOffloader
│
├── Agentic Layer
│   ├── Long-horizon state machine (from TradingAgents / JuniorAGI)
│   ├── Tool allow-list + sandbox
│   └── Continuous eval + red-team harness (from vault)
│
└── Safety Fallback Path
    └── Constrained / smaller BitNet model or explicit refusal
```

## 4. Safety Classifier (Fable-Inspired)

Inspired by Fable 5’s external classifiers:
- Detect high-risk categories (cyber misuse, bio/chem, model distillation attempts, etc.)
- On detection → route to a more constrained path or refuse with clear message
- Log and surface the classification for transparency (matches Fable’s “users are informed” pattern)
- Implemented as a lightweight classifier that can itself be BitNet or rules + small model

This lives in `safety/classifier.py` and is wired into the JuniorLLM entrypoint.

## 5. Relationship to JuniorPortal-K3

- Both share the same BitNet core, offloader, TDA, and production-loop infrastructure.
- Kimi path emphasizes sparse MoE scale + long context efficiency.
- Fable path emphasizes agentic reliability, safety classifiers, and high-consistency long-horizon behavior.
- A model router inside JuniorLLM can choose “Kimi-style”, “Fable-style”, or hybrid based on task (coding vs trading vs general agentic).

## 6. Development Principles

- Never delete existing Kimi adaptation files or other ecosystem repos.
- All new work is additive.
- Use the same production loop pattern.
- Target edge-first (M4 / van) from day one.
- Integrate vault security notes and risk assessment templates.

## 7. Immediate Scaffolding

- This architecture document
- Safety classifier skeleton
- Behavior card for Fable-style JuniorLLM
- Adaptation loop
- Push of key files into the existing JuniorLLM GitHub repository (additive)

*Fable-class agentic reliability + BitNet edge sovereignty.*
