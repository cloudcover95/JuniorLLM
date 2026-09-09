# JuniorAstra Portal

Open-source **Astra** capabilities adapted into JuniorCloud BitNet — same slot as JuniorGemma-4 and JuniorLLM-Fable.

## What we learned from (open)

| Source | License | What we take |
|--------|---------|----------------|
| [matrixorigin/Astra](https://github.com/matrixorigin/Astra) | Apache-2.0 | Durable Work, Server/Runner split, ContextPipe budgets, checkpoints, trace/rollback |
| ASTRA agentic-trajectory research (14B/32B thinking papers) | research | Maker/checker trajectory shape, long-horizon tool loops |
| ICLR 2026 Astra world-model | MIT weights (video) | Action-conditioned rollout *idea* only — not pulled into this LLM port |

## What we do **not** take

**GPT-6 Astra / OpenAI Astra is closed.** No weights, no API wrapper, no `gpt-6-astra` client. JuniorAstra is a local runtime + BitNet substrate that copies the *open* agent-runtime contract.

## Port contract (like Gemma / Fable)

| Piece | Role |
|-------|------|
| **Work** | Long-lived goal with stable `work_id` beyond one chat turn |
| **ContextPipe** | Budgeted context assembly (tokens in, provenance out) |
| **Runner** | Edge execution inside the Junior trust boundary (BitNet + TDA + Fable gate) |
| **Checkpoint** | Pause / resume overnight via `data/astra/works/` |
| **Trace** | Append-only events for rollback |

LLM backend is **brought by us**: FieldCore / Gemma4 Q4 / BitNet-2B4T — same as the open Astra README (“bring your own LLM”).

## Status

Runnable kernel + tests. No 14B/32B download required to exercise the portal.
