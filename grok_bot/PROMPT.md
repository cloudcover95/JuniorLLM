# ROUTINE — JuniorCloud overnight (Automations)

Standing job, not chat. Owner cloudcover95. TZ America/Denver.

End objective: live production-grade beta of the local LLM suite, then JuniorOS overlay (`docs/BETA_TO_OS.md`).

Input: GitHub `cloudcover95/JuniorLLM` (AGENTS.md, docs/BETA_TO_OS.md, grok_bot/LAST_RECEIPT.md).
Output: one additive commit and an updated `grok_bot/LAST_RECEIPT.md`.
Approval: stop if plan >8 files (`needs_split`).
Failure: GitHub down → `unavailable`. Do not invent. Do not device-login. Do not PAT.

## Each night
1. Read LAST_RECEIPT `next_smallest_slice` and BETA_TO_OS backlog.
2. Implement exactly that slice (additive).
3. Prefer tests: test_junior_aie, test_astra_reason, juniorctl health.
4. Commit with `port: <JuniorPort>`.
5. Rewrite LAST_RECEIPT. Name the following backlog line.

Never delete. Never >8GB. Never GPT-6 Astra client. Never MP/KAYA scrape. bitnetd loopback only.
Do not contact anyone.
