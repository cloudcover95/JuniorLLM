# How much one automation day can do

Measured 2026-09-09 America/Denver on task `JuniorCloud overnight build`.

| run | title | wall |
|-----|--------|------|
| 03:11 | Overlay installer | 3.0 min |
| 03:11 | Overlay installer (overlap) | 3.1 min |
| 03:18 | **A8 evals harness** | **8.4 min** |

A8 = SUCCESS. `COMPILED_BACKLOG` A8 is `done`. Next pointer in receipt: **A9 STATE schema**.

## Throughput math

- One run = one additive slice (≤8 files) + receipt + optional tests.
- Observed 3–9 minutes. Budget **15 minutes** including GitHub lag.
- Two jobs started within 21 seconds tonight and both wrote overlay files. That is the failure mode.
- 24 hourly jobs would spend most of the day colliding on `LAST_RECEIPT.md`.

**Max useful schedule: 4 slots / day, ≥6 hours apart.**

| slot | local | role |
|------|-------|------|
| 0 | 01:00 | primary night (existing) |
| 1 | 07:00 | morning backfill |
| 2 | 13:00 | midday |
| 3 | 19:00 | evening |

Expected: **4 slices / day**, ~12–36 min model time, 4 commits.
Open+skeleton rows now ~12 → about **3 days** if every slot ships.
If a slot sees `LAST_RECEIPT.run_at` younger than 45 minutes → `skip_overlap`, no commit.

Do not add hourly 01–19. Quality drops and GitHub secondary-rate limits start to matter before raw minutes do.
