# Test roadmap — local + high-quant models (2026 H2)

## Tier 0 — always on (van / no lab)
- JuniorStoneField engine suite (25 tests)
- Guardrail pipeline unit tests (deny private-land publish, deny eval/exec in skills)
- Self-prompt schema validation
- Port registry loads without network

## Tier 1 — edge BitNet native
| Artifact | Quant | Target |
|----------|-------|--------|
| microsoft/bitnet-b1.58-2B-4T-gguf | I2_S / TQ1_0 | CPU bitnet.cpp tok/s + refusal on closed beta ingest |
| 1bitLLM bitnet_b1_58 family | TQ1_0 / TQ2_0 | FieldCore condition/access F1 vs cue labels |
| JuniorBitNetFieldCore | ternary 32-d | embed determinism, disagreement rise |

Eval: perplexity proxy optional; **required** = covenant gate + TDA recommendation vocab match.

## Tier 2 — high-quant general local (not BitNet-native)
| Artifact | Quant | When |
|----------|-------|------|
| Gemma 4 4B (Apache) | Q4_K_M / MLX 4-bit | interactive JuniorGemma portal |
| Qwen 3.x 8B–27B | Unsloth dynamic GGUF / 1-bit *post-quant* | quality vs native BitNet honesty test |
| Kimi K3 edge prune | expert-offload | only if weights fit device; never 1.5TB download |

Note: post-hoc 1-bit of a dense model ≠ BitNet-native. Roadmap treats them as **separate ports**.

## Tier 3 — agent overnight
- Overnight job writes STATE.md + receipt JSON
- Maker/checker: TDA gate must pass or job status=`review_needed`
- Memory topics never store health/beliefs unless `include_sensitive=true`
- Swift rail compiles Guardrail.swift (macOS CI later)
- Linux `bitnetd` health endpoint on localhost only

## Tier 4 — OS-world style (future BitNet Linux)
Do **not** claim OSWorld scores. Track:
- local file create/read via Files ledger
- skill load + hash pin
- GNSS NMEA ingest (already tested)
- no outbound vendor SDK
