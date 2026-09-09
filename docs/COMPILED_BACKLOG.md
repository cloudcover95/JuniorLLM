# Compiled backlog (do not drop old checklists)

Overnight reads **this file** plus `LAST_RECEIPT.next_smallest_slice`.
Capacity: `docs/AUTOMATION_CAPACITY.md` — 4 slots/day max.

Status: `done` | `skeleton` | `open`

## A — live beta suite

| id | item | status |
|----|------|--------|
| A1 | juniorctl health / ask / night / security | done |
| A2 | junior_aie 15-piece + tests | done |
| A3 | Astra runtime + AstraReason + Fable + Gemma loader stubs | done |
| A4 | Guardrail deny eval/exec + private-land publish | done |
| A5 | Port registry no-network | done |
| A6 | StoneField engines + Climbs stdlib probe | done |
| A7 | Missing package `__init__.py` if import breaks | open |
| A8 | `evals/` harness (FLOPS proxy + accuracy contract) | done |
| A9 | STATE.md + LAST_RECEIPT.json schema | done |

## B — local LLM suite

| id | item | status |
|----|------|--------|
| B1 | FieldCore ternary | done |
| B2 | BitNet-2B4T on-disk only | skeleton |
| B3 | Gemma4 Q4_K_M / MLX loader | skeleton |
| B4 | Qwen-local cap 8GB | skeleton |
| B5 | Kimi-edge prune — never 1.5TB | skeleton |
| B6 | Fable SafetyClassifier on router | done |
| B7 | NightTernary DreamMesh | done |
| B8 | Multi-portal production loop file | done |
| B9 | Prompt registry + flywheel receipts | done |

## C — JuniorOS + container security

| id | item | status |
|----|------|--------|
| C1 | os-release + bitnetd loopback | done |
| C2 | install-overlay.sh | done |
| C3 | seccomp + systemd harden | done |
| C4 | rootless OCI example, no docker.sock | open |
| C5 | bitnetd real I2_S when GGUF present | open |
| C6 | Files ledger create/read | open |
| C7 | skill load + hash pin | open |

## D — older rails

| id | item | status |
|----|------|--------|
| D1 | rails/swift Guardrail.swift | done |
| D2 | Obsidian vault bridge | done |
| D3 | Grok Bot AGENTS + skills | done |
| D4 | maker/checker | done |
| D5 | GNSS/NMEA | done |
| D6 | gym_internal notes not leaked on public ask | open |

## Night rule

One slice. If LAST_RECEIPT is <45 min old → skip_overlap.
Fill skeleton before new tree. Rewrite STATE.md + LAST_RECEIPT.md + LAST_RECEIPT.json.
