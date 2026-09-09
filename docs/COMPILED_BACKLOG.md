# Compiled backlog (do not drop old checklists)

Overnight reads **this file** plus `LAST_RECEIPT.next_smallest_slice`.
New work still aims at live beta → local LLM suite → JuniorOS overlay.
Old lists are not retired; they are rows below.

Sources merged:
- early AGENTS / grok_bot checklist (init files, bitnetd, Gemma path, multi-portal, low_confidence stop)
- `docs/TEST_ROADMAP_2026.md` tiers 0–4
- `docs/BETA_TO_OS.md`
- `rails/linux/ROADMAP.md` + `CONTAINER_SECURITY.md`
- Swift rail, Obsidian, AIE 15-piece, custom LLM ports

Status: `done` | `skeleton` | `open`

## A — live beta suite (must stay green)

| id | item | status |
|----|------|--------|
| A1 | juniorctl health / ask / night / security | done |
| A2 | junior_aie 15-piece + tests | done |
| A3 | Astra runtime + AstraReason + Fable + Gemma loader stubs | done |
| A4 | Guardrail deny eval/exec + private-land publish | done |
| A5 | Port registry no-network | done |
| A6 | StoneField engines + Climbs stdlib probe | done |
| A7 | Missing package `__init__.py` if import breaks | open |
| A8 | `evals/` harness (FLOPS proxy + accuracy contract) | skeleton |
| A9 | `STATE.md` written each night | skeleton |

## B — local LLM suite (old + new ports)

| id | item | status |
|----|------|--------|
| B1 | FieldCore ternary | done |
| B2 | BitNet-2B4T on-disk only (`ports/ON_DEVICE.md`) | skeleton |
| B3 | Gemma4 Q4_K_M / MLX loader | skeleton |
| B4 | Qwen-local cap 8GB | skeleton |
| B5 | Kimi-edge prune — never 1.5TB | skeleton |
| B6 | Fable SafetyClassifier on router | done |
| B7 | NightTernary DreamMesh | done |
| B8 | Multi-portal production loop file | done |
| B9 | Prompt registry + flywheel receipts | done |

## C — JuniorOS overlay + container security

| id | item | status |
|----|------|--------|
| C1 | os-release.junior + bitnetd.service loopback | done |
| C2 | install-overlay.sh | done |
| C3 | seccomp + systemd harden + juniorctl security | done |
| C4 | rootless OCI example, no docker.sock | open |
| C5 | bitnetd real I2_S when GGUF present | open |
| C6 | Files ledger create/read (TEST_ROADMAP T4) | open |
| C7 | skill load + hash pin | open |

## D — rails still owed from older lists

| id | item | status |
|----|------|--------|
| D1 | rails/swift Guardrail.swift | done |
| D2 | Obsidian vault bridge folder | done |
| D3 | Grok Bot AGENTS + skills | done |
| D4 | maker/checker in overnight | done |
| D5 | GNSS/NMEA (Climbs navmesh) | done |
| D6 | gym_internal notes not leaked on public ask | open |

## Night rule

Take the first `open` or `skeleton` row after LAST_RECEIPT's pointer.
Fill a skeleton before opening a new tree.
Additive only. One slice. Rewrite LAST_RECEIPT + STATE.md.
