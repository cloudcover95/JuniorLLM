---
local: true
date: 2026-11-05
wave: "2026-11-05 10:05 (bench 10:55, ran ~11:16–11:37 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` cold B=16/32/64 Deck/SBC rate-cap × modeled I/O queue-depth K sweep pair vs int8)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, ask, cold, rate-cap, queue-depth, readahead, deck, sbc, bench, b16, b32, b64, viable, k-sweep]
---

# juniorctl ask, cold B = 16 / 32 / 64 under Deck/SBC rate caps: modeled queue depth K does not collapse the pair lead

## Question

The 05:55 note ([[2026-11-05_0555_ask-b16-b32-b64-deck-sbc-rate-cap-x-memory-cap-cold-pair-vs-int8]]) held the same ~4.3x to 4.9x I/O-bound band under rate-cap × memory-cap and asked whether a real Deck/SBC media path (device queueing / readahead) collapses that lead toward the uncapped ~1.8x to 2.1x neighborhood. This box has no physical eMMC/SD, so this 10:55 run answers the queueing/readahead half by sweeping modeled requests-in-flight K ∈ {1, 2, 8, 16} under the same 100 / 150 MB/s Deck/SBC rate caps (ternary compute / ask path; N = 2M, same as the 14:55 rate-cap-alone note).

## Method

Synthetic 2,000,000-row x 384-dim index built with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), streamed to two flat files under the scratch dir and fsynced. Int8 file is 768,000,000 B; pair file is 156,000,000 B (4.92x smaller). Same device model as the 14:55 / 05:55 rate benches: fixed 5 ms latency per request, K requests in flight (semaphore), bytes then reserved one-at-a-time at R MB/s. Chunk size 8,192 rows; 8 worker threads; OpenBLAS / OMP pinned to 1. Scoring per chunk is int8 cast→float32 GEMM or pair `np.take(V10,…)` unpack→float32 GEMM, then chunked top-10 merge. Cold only: `posix_fadvise(DONTNEED)` before every timed batch. Batches B ∈ {16, 32, 64} slice the 64-query build matrix (rotating offset per repeat). ≥5 timed cold batches per cell; format order reversed on odd repeats. Every timed batch checked against an uncapped single-thread int8 reference (exact set equality). Rates: 100 MB/s (SBC SD / slow Deck SD) and 150 MB/s (Deck eMMC-ish). K ∈ {1, 2, 8, 16} as a first-class axis (printed in the results table). numpy 2.5.4, 8 vCPU shared box, MemAvailable ~6.1 GB at start, python3 and numpy only, no network, nothing bound except the local files. A 131,072-row smoke at 100 MB/s / K = 1,2,8 / B = 16,32 passed first (top-10 exact). Script (`bench.py`, scratch `/workspace/bench1055_1105/`) imports `build`, `V10`, `D`, `PW`, `TOP`, `Dev` from the 2026-10-30 14:55 scratch. Commands, data bins deleted afterwards:

```
python3 bench.py 131072 202611051055 1 100 5 8 1,2,8 8192 16,32
python3 bench.py 2000000 202611051055 5 100,150 5 8 1,2,8,16 8192 16,32,64
```

## Results

Median ms per B-batch, p10 to p90 in brackets. Top-10 matched the uncapped single-thread int8 reference exactly in every timed batch. "pair vs int8" is int8 median / pair median. Observed read rates are derived from file bytes / median wall time. Latency floor ≈ chunks × 5 ms / K (≈ 1,221 / 611 / 153 / 76 ms at K = 1 / 2 / 8 / 16).

### Rate = 100 MB/s (SBC SD / slow Deck SD)

| K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|
| 1 | 16 | 9011.3 (9009.2-9016.7) | 2893.8 (2887.9-2896.5) | **3.11x** | 85 | 54 | exact |
| 1 | 32 | 9009.5 (8997.3-9015.0) | 2885.6 (2878.7-2901.5) | **3.12x** | 85 | 54 | exact |
| 1 | 64 | 9021.6 (9017.0-9023.0) | 2898.3 (2895.0-2898.9) | **3.11x** | 85 | 54 | exact |
| 2 | 16 | 7690.4 (7689.9-7690.9) | 1572.8 (1572.4-1573.9) | **4.89x** | 100 | 99 | exact |
| 2 | 32 | 7694.3 (7693.7-7694.6) | 1576.7 (1575.8-1584.4) | **4.88x** | 100 | 99 | exact |
| 2 | 64 | 7699.4 (7698.6-7700.1) | 1582.0 (1581.6-1582.6) | **4.87x** | 100 | 99 | exact |
| 8 | 16 | 7689.8 (7689.6-7690.0) | 1573.5 (1572.4-1574.2) | **4.89x** | 100 | 99 | exact |
| 8 | 32 | 7693.1 (7692.7-7693.4) | 1575.5 (1574.9-1576.4) | **4.88x** | 100 | 99 | exact |
| 8 | 64 | 7700.3 (7698.9-7701.8) | 1583.2 (1581.7-1585.4) | **4.86x** | 100 | 99 | exact |
| 16 | 16 | 7690.5 (7690.2-7694.7) | 1573.3 (1573.0-1573.6) | **4.89x** | 100 | 99 | exact |
| 16 | 32 | 7693.3 (7692.9-7694.1) | 1576.6 (1576.4-1578.4) | **4.88x** | 100 | 99 | exact |
| 16 | 64 | 7700.6 (7700.1-7702.1) | 1583.2 (1582.7-1584.7) | **4.86x** | 100 | 99 | exact |

### Rate = 150 MB/s (Deck eMMC-ish)

| K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|
| 1 | 16 | 6450.9 (6446.6-6458.7) | 2366.7 (2362.8-2372.6) | **2.73x** | 119 | 66 | exact |
| 1 | 32 | 6449.6 (6443.7-6457.0) | 2372.1 (2365.9-2373.9) | **2.72x** | 119 | 66 | exact |
| 1 | 64 | 6462.0 (6456.3-6468.6) | 2375.3 (2371.9-2377.0) | **2.72x** | 119 | 66 | exact |
| 2 | 16 | 5131.9 (5131.5-5132.7) | 1189.8 (1185.5-1198.4) | **4.31x** | 150 | 131 | exact |
| 2 | 32 | 5135.3 (5134.3-5135.9) | 1192.8 (1184.7-1197.1) | **4.31x** | 150 | 131 | exact |
| 2 | 64 | 5143.0 (5141.5-5143.6) | 1201.0 (1199.1-1202.1) | **4.28x** | 149 | 130 | exact |
| 8 | 16 | 5131.9 (5131.2-5132.5) | 1053.7 (1053.6-1054.9) | **4.87x** | 150 | 148 | exact |
| 8 | 32 | 5135.5 (5134.5-5136.3) | 1057.1 (1055.7-1058.4) | **4.86x** | 150 | 148 | exact |
| 8 | 64 | 5142.0 (5141.3-5143.3) | 1063.8 (1062.7-1066.7) | **4.83x** | 149 | 147 | exact |
| 16 | 16 | 5131.9 (5131.3-5132.3) | 1053.6 (1053.0-1070.7) | **4.87x** | 150 | 148 | exact |
| 16 | 32 | 5135.2 (5133.8-5136.0) | 1055.8 (1055.0-1056.7) | **4.86x** | 150 | 148 | exact |
| 16 | 64 | 5141.2 (5140.8-5141.3) | 1062.1 (1061.3-1065.8) | **4.84x** | 149 | 147 | exact |

Raising modeled queue depth does **not** collapse the lead toward uncapped ~1.8x to 2.1x. At 100 MB/s, K = 1 is latency-bound on the pair path (~3.11x to 3.12x); from K = 2 onward both formats sit on their byte floors and the ratio holds at ~4.86x to 4.89x (same band as rate-cap-alone / rate-cap × memory-cap). At 150 MB/s, K = 1 is ~2.72x to 2.73x; K = 2 recovers ~4.28x to 4.31x; K = 8 and K = 16 push further to ~4.83x to 4.87x as the pair path escapes its remaining latency floor toward the 4.92x size ratio. K = 1 → K = 16 therefore *widens* the lead, not collapses it. Memory remains 4.92x smaller. (In flight above 8 cannot exceed the 8 worker threads, so K = 16 equals K = 8 in this harness.)

## Verdict

**Viable.** Under Deck/SBC-class rate caps (100 / 150 MB/s) with 5 ms latency and 8,192-row chunks at N = 2,000,000, raising modeled I/O queue depth K from 1 to 16 does **not** collapse the cold pair-code vs int8 lead toward the uncapped ~1.8x to 2.1x neighborhood. At K = 16 the lead is **4.86x to 4.89x** at 100 MB/s and **4.84x to 4.87x** at 150 MB/s for B = 16 / 32 / 64 (exact top-10 every batch), vs K = 1 ratios of ~3.11x and ~2.72x. Higher queue depth recovers / holds the I/O-bound ~4.8x to 4.9x band. Memory is also 4.92x smaller. Bounded to this synthetic / modeled-device setup; real Deck/SBC media may still differ, but the modeled queueing/readahead half clears the 1.5x bar with margin and answers the 05:55 Next question's queue-depth half in the negative (no collapse).

Caveats: synthetic embeddings; modeled device (latency + rate semaphore with K-slot), not a physical SD/eMMC; N = 2M; numpy / interpreter overhead; shared 8 vCPU box; one seed, five cold repeats per cell; K = 16 is capped by 8 workers so it matches K = 8 behavior here.

## Pair files this wave

- No pair file for the 10:05 wave reached this run (Chief both pairs no gap; Trit Pack / Winsor / Side-Commit / Handshake correctly silent; roadmap blobs still JuniorHome ROADMAP 4efb666 / JuniorLLM BETA_TO_OS bac6923). JuniorLLM tip before this push was [22df49f](https://github.com/cloudcover95/JuniorLLM/commit/22df49f) (05:55 vault findings). The `ask` side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Modeled queue depth under Deck/SBC rate caps does not collapse the lead. Does a real physical eMMC/SD media path (or a host with device-level readahead / NCQ that this semaphore cannot express) still hold the ~4.8x to 4.9x band, or does host page-cache + block-layer readahead on warm runs collapse cold-only gains once the first pass fills cache?

close: 1644d691cc922790 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-05_0555)
