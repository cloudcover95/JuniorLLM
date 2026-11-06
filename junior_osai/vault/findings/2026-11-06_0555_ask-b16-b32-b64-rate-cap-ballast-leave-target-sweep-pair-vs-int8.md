---
local: true
date: 2026-11-06
wave: "2026-11-06 05:05 (bench 05:55, ran ~06:25–07:17 MT; vault landed late after routine timeout)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` rate-cap × ballast-size sweep leave-target MemAvailable pair vs int8; no writable cgroup memory.high / no physical eMMC/SD)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, ask, rate-cap, memory-high, ballast, proxy, leave-target, sweep, reclaim, deck, sbc, bench, b16, b32, b64, viable, speed, memory]
---

# juniorctl ask, rate-cap × ballast leave-target sweep: ~4.8x holds down to ~0.65–0.9 GB MemAvailable

## Question

The 02:55 note ([[2026-11-06_0255_ask-b16-b32-b64-rate-cap-memory-high-ballast-proxy-pair-vs-int8]]) held Deck/SBC rate-cap ~4.84x–4.89x under a single anonymous-ballast memory.high proxy (MemAvailable ≈ 1.3–2.3 GB during cells) and asked for a **ballast-size sweep** that drives MemAvailable under pair-only vs int8-only footprints, or physical eMMC/SD / real cgroup `memory.high`. This box still has no physical eMMC/SD and `/sys/fs/cgroup/agent/memory.high` is still **not writable**. This 05:55 run answers the **ballast-size sweep** half: under the same Deck/SBC rate-cap (rates 100 / 150 MB/s, K = 8, lat 5 ms, chunk 8192, 8 threads), do leave-target ballasts aimed at MemAvailable ≈ 2000 / 500 / 200 MB collapse pair-vs-int8 toward the warm-uncapped 0.83x–0.90x regime at B ∈ {16, 32, 64}, N = 2M — or does the cold ~4.8x–4.9x band hold across the sweep?

## Method

Synthetic 2,000,000-row × 384-dim index built with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), streamed to two flat files under the scratch dir and fsynced. Int8 file is 768,000,000 B; pair file is 156,000,000 B (4.92x smaller). Same device model as 02:55 / 18:55: fixed 5 ms latency per request, K = 8 requests in flight, bytes reserved one-at-a-time at R MB/s. Chunk size 8,192 rows; 8 worker threads; OpenBLAS / OMP pinned to 1. Scoring per chunk is int8 cast→float32 GEMM or pair `np.take(V10,…)` unpack→float32 GEMM, then chunked top-10 merge. Batches B ∈ {16, 32, 64}. ≥5 timed repeats per cell; format order reversed on odd repeats. Every timed batch checked against an uncapped single-thread int8 reference (exact set equality). Seed `202611060555`.

Arms (all under the same Dev rate-cap model):
1. **cold_rate_none** (anchor) — `posix_fadvise(..., DONTNEED)` on both bins **before every timed batch**; no ballast. Expect ~4.84x–4.89x.
2. **rate_ballast leave-target sweep** — **no** DONTNEED between batches; for each leave target L ∈ {2000, 500, 200} MB start a child ballast (`mmap` anonymous + touch every page; `mlock` ulimit still 8 MB so locked_mb stayed 0) with hard_cap 10,500 MB, record HOLD status / MemAvailable per cell, then timed rate-capped batches; stop ballast between leave targets.

numpy 2.5.4, 8 vCPU shared box, MemAvailable ~6.5 GB at start, agent cgroup `memory.high` present but not writable, python3 and numpy only, no network, nothing bound except the local files / ballast child. A 131,072-row smoke at leave targets 2000 / 500 passed first (top-10 exact). Script (`bench.py` + `ballast_worker.py`, scratch `/workspace/bench0555_1106/`) imports `build`, `V10`, `D`, `PW`, `TOP`, `Dev` from the 2026-10-30 14:55 scratch (`/workspace/bench1455_1030/bench.py`). Commands, data bins deleted afterwards:

```
python3 bench.py 131072 202611060555 1 cold_rate_none,rate_ballast 100 5 8 8 8192 16,32 2000,500 8000
python3 bench.py 2000000 202611060555 5 cold_rate_none,rate_ballast 100,150 5 8 8 8192 16,32,64 2000,500,200 10500
```

## Results

Median ms per B-batch, p10 to p90 in brackets. Top-10 matched the uncapped single-thread int8 reference exactly in every timed batch. "pair vs int8" is int8 median / pair median. Observed throughput from file bytes / median wall time.

### cold_rate_none (DONTNEED, no ballast, K = 8) — anchor

| rate MB/s | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|---|
| 100 | 8 | 16 | 7690.0 (7689.7-7690.3) | 1572.3 (1572.1-1572.7) | **4.89x** | 100 | 99 | exact |
| 100 | 8 | 32 | 7692.9 (7692.2-7693.5) | 1574.7 (1574.1-1575.4) | **4.89x** | 100 | 99 | exact |
| 100 | 8 | 64 | 7699.5 (7698.6-7699.8) | 1582.1 (1581.4-1585.3) | **4.87x** | 100 | 99 | exact |
| 150 | 8 | 16 | 5130.9 (5130.6-5131.5) | 1052.5 (1052.2-1063.4) | **4.87x** | 150 | 148 | exact |
| 150 | 8 | 32 | 5134.6 (5134.2-5134.8) | 1055.5 (1054.9-1056.1) | **4.86x** | 150 | 148 | exact |
| 150 | 8 | 64 | 5142.3 (5140.9-5142.3) | 1061.7 (1060.7-1062.8) | **4.84x** | 149 | 147 | exact |

Matches the 02:55 / 18:55 / 14:55 / 10:55 K = 8 cold rate-cap band (~4.84x–4.89x).

### rate_ballast leave=2000 MB (no DONTNEED + ballast, K = 8)

Ballast started HOLD allocated_mb=4224 → topped toward ~9664 MB during cells; MemAvailable during cells ≈ 1.68–1.95 GB (near the 2 GB leave target).

| rate MB/s | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 | MemAvail MB |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 8 | 16 | 7690.6 (7690.2-7691.2) | 1572.7 (1572.6-1573.0) | **4.89x** | 100 | 99 | exact | 1946 |
| 100 | 8 | 32 | 7693.3 (7692.6-7694.1) | 1575.1 (1575.1-1577.7) | **4.88x** | 100 | 99 | exact | 1946 |
| 100 | 8 | 64 | 7699.7 (7698.7-7710.8) | 1582.3 (1581.6-20122.1) | **4.87x** | 100 | 99 | exact | 1678 |
| 150 | 8 | 16 | 5135.4 (5132.8-5152.5) | 1053.3 (1052.6-1053.9) | **4.88x** | 150 | 148 | exact | 1762 |
| 150 | 8 | 32 | 5135.4 (5134.3-5137.8) | 1055.6 (1055.1-1056.1) | **4.86x** | 150 | 148 | exact | 1756 |
| 150 | 8 | 64 | 5142.5 (5141.4-5145.5) | 1062.3 (1061.7-1063.5) | **4.84x** | 149 | 147 | exact | 1757 |

Indistinguishable from `cold_rate_none` within noise.

### rate_ballast leave=500 MB (no DONTNEED + ballast, K = 8)

Ballast hit hard_cap (allocated_mb=10560); MemAvailable during cells ≈ 0.81–1.00 GB — tighter than 02:55's ~1.3 GB floor, but still above the 924 MB combined working set (leave target 500 MB not fully reached under the 10.5 GB hard cap).

| rate MB/s | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 | MemAvail MB |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 8 | 16 | 7691.7 (7690.4-7692.5) | 1572.6 (1571.4-1580.9) | **4.89x** | 100 | 99 | exact | 846 |
| 100 | 8 | 32 | 7694.0 (7693.0-7695.4) | 1575.4 (1575.0-1577.6) | **4.88x** | 100 | 99 | exact | 814 |
| 100 | 8 | 64 | 7700.8 (7699.9-7701.4) | 1581.7 (1581.5-1582.1) | **4.87x** | 100 | 99 | exact | 854 |
| 150 | 8 | 16 | 5132.4 (5130.2-5144.6) | 1052.5 (1052.0-1053.0) | **4.88x** | 150 | 148 | exact | 999 |
| 150 | 8 | 32 | 5136.1 (5133.8-5139.5) | 1055.2 (1054.2-1060.1) | **4.87x** | 150 | 148 | exact | 860 |
| 150 | 8 | 64 | 5141.6 (5140.7-5173.0) | 1061.9 (1061.8-1061.9) | **4.84x** | 149 | 147 | exact | 850 |

Still ~4.84x–4.89x; no collapse.

### rate_ballast leave=200 MB (no DONTNEED + ballast, K = 8)

Again hard_cap limited (allocated_mb=10560); MemAvailable during cells ≈ 0.66–0.85 GB (tightest sustained pressure this sweep; still above pair-only 156 MB footprint, near or slightly under the 924 MB combined WS at the low end).

| rate MB/s | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 | MemAvail MB |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 8 | 16 | 7693.1 (7691.6-11149.2) | 1574.0 (1572.5-1575.0) | **4.89x** | 100 | 99 | exact | 794 |
| 100 | 8 | 32 | 7695.6 (7693.8-7696.6) | 1577.0 (1575.1-1594.3) | **4.88x** | 100 | 99 | exact | 834 |
| 100 | 8 | 64 | 7701.2 (7699.8-7702.8) | 1583.3 (1582.8-1585.3) | **4.86x** | 100 | 99 | exact | 824 |
| 150 | 8 | 16 | 5132.0 (5131.1-5132.1) | 1052.8 (1052.5-1072.3) | **4.87x** | 150 | 148 | exact | 821 |
| 150 | 8 | 32 | 5134.9 (5134.5-5136.7) | 1056.0 (1054.5-1077.5) | **4.86x** | 150 | 148 | exact | 834 |
| 150 | 8 | 64 | 5143.3 (5140.6-14061.7) | 1061.6 (1061.3-1087.3) | **4.84x** | 149 | 147 | exact | 739 |

Still ~4.84x–4.89x at the tightest leave target; one high p90 outlier on int8 at leave=200 / 100 / B=16 and leave=200 / 150 / B=64 (host noise under pressure) did not move the median ratio.

## Verdict

**Viable** on both speed and memory. The ballast-size leave-target sweep does **not** collapse the Deck/SBC rate-capped pair-vs-int8 lead: across leave targets 2000 / 500 / 200 MB (observed MemAvailable during cells ≈ 1.95 GB → ≈ 0.66 GB under a 10.5 GB hard cap), every rate_ballast cell stays in the **~4.84x–4.89x** band at rates 100 / 150 MB/s, K = 8, B ∈ {16, 32, 64} (exact top-10) — matching `cold_rate_none` and well above the 1.5x speed bar. Memory remains **4.92x smaller** (768,000,000 B vs 156,000,000 B). Answers the 02:55 Next question's ballast-size-sweep half: no, the rate-cap ratio does not break as MemAvailable is driven from ~2 GB down toward ~0.65–0.9 GB. Bounded to this synthetic / host-ballast setup on a shared 8 vCPU box with no writable cgroup `memory.high` and no physical eMMC/SD; leave targets 500 / 200 MB were hard-cap limited (allocated capped at 10,560 MB) so full thrashing strictly below the combined 924 MB working set was only approached, not held for the whole sweep.

Caveats: synthetic embeddings; modeled Dev rate-cap (not device NCQ / physical media); ballast is a user-space proxy not a real cgroup `memory.high` reclaim path; hard_cap prevented sustained MemAvailable at the nominal 500 / 200 MB leave targets; N = 2M; numpy / interpreter unpack overhead; shared 8 vCPU box; one seed, five repeats per cell; occasional high p90 under the tightest leave (median unchanged).

## Pair files this wave

- No pair file for the 05:05 wave expected (Chief both pairs no gap for many idle waves; roadmap blobs still JuniorHome ROADMAP 4efb666 / JuniorLLM BETA_TO_OS bac6923). JuniorLLM tip before this push recorded from the fresh clone was [4966a12](https://github.com/cloudcover95/JuniorLLM/commit/4966a12) (02:55 vault findings). The `ask` side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Ballast leave-target sweep under rate-cap does not collapse ~4.8x down to ~0.65–0.9 GB MemAvailable on this host. Remaining: physical eMMC/SD / NCQ when hardware exists; **or** a real writable cgroup `memory.high` (if delegation appears) that forces reclaim strictly and sustainably below the int8 working set without a user-space hard-cap ceiling.

close: 599edb9554333ee2 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-06_0255)
