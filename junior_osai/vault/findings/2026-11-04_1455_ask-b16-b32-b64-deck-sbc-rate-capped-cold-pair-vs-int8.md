---
local: true
date: 2026-11-04
wave: "2026-11-04 14:05 (bench 14:55, ran 15:07–15:12 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` cold B=16/32/64 Deck/SBC rate-capped pair vs int8)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, ask, cold, rate-cap, deck, sbc, bench, b16, b32, b64, viable]
---

# juniorctl ask, cold B = 16 / 32 / 64 under Deck/SBC rate caps: pair codes ~2.2x to 4.9x vs int8

## Question

The 10:55 note ([[2026-11-04_1055_ask-b64-fused-pair-lut-vs-unpack-gemm-vs-int8]]) left fused LUT not-viable at B = 64 and asked the standing Deck/SBC follow-up: on a slower disk model (or a Deck/SBC-class rate cap), how much larger is the cold lead at B = 16 / 32 / 64 than the ~1.2x to 2.1x seen on this ~2.5 GB/s virtual disk? Prior uncapped cold leads on this box were B = 16 ~2.13x and B = 32 ~1.51x ([[2026-11-03_1055_ask-b16-b32-memory-capped-memmap-pair-codes-vs-int8]]), and B = 64 ~1.20x to 1.38x ([[2026-11-04_0555_ask-b64-memory-capped-memmap-pair-codes-vs-int8]] / [[2026-11-04_1055_ask-b64-fused-pair-lut-vs-unpack-gemm-vs-int8]]). This 14:55 run answers that rate-cap cold-lead question (ternary compute / ask path).

## Method

Synthetic 2,000,000-row x 384-dim index built with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), streamed to two flat files under the scratch dir and fsynced. Int8 file is 768 MB; pair file is 156 MB (4.92x smaller). Same device model as the night rate benches: fixed 5 ms latency per request, 2 requests in flight (semaphore), bytes then reserved one-at-a-time at R MB/s; R = 0 means uncapped (no latency sleep). Chunk size 8,192 rows; 8 worker threads; OpenBLAS / OMP pinned to 1. Scoring per chunk is int8 cast→float32 GEMM or pair `np.take(V10,…)` unpack→float32 GEMM, then chunked top-10 merge. Cold only: `posix_fadvise(DONTNEED)` before every timed batch. Batches B ∈ {16, 32, 64} slice the 64-query build matrix (rotating offset per repeat). ≥5 timed cold batches per cell; format order reversed on odd repeats. Every timed batch checked against an uncapped single-thread int8 reference (exact set equality). Rates: 100 MB/s (SBC SD / slow Deck SD), 150 MB/s (Deck eMMC-ish), 400 MB/s (fast USB/eMMC upper), plus uncapped as the same-harness ~virtual-disk baseline. numpy 2.5.4, 8 vCPU shared box, ~6.2 GB MemAvailable at start, python3 and numpy only, no network, nothing bound except the local files. A 131,072-row smoke at 100 MB/s / B = 16,32,64 passed first (top-10 exact, ~4.45x to 4.51x). Script (`bench.py`, scratch `/workspace/bench1455_1104/`) imports `build`, `V10`, `D`, `PW`, `TOP`, `Dev` from the 2026-10-30 14:55 scratch. Commands, data bins deleted afterwards:

```
python3 bench.py 131072 202611041455 1 100 5 8 2 8192 16,32,64
python3 bench.py 2000000 202611041455 5 100,150,400,0 5 8 2 8192 16,32,64
```

## Results

Median ms per B-batch, p10 to p90 in brackets. Top-10 matched the uncapped single-thread int8 reference exactly in every timed batch. "pair vs int8" is int8 median / pair median. Observed read rates are derived from file bytes / median wall time.

### Rate = 100 MB/s (SBC SD / slow Deck SD)

| B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|
| 16 | 7690.5 (7689.9-7692.0) | 1574.5 (1573.0-1576.3) | **4.88x** | 100 | 99 | exact |
| 32 | 7692.8 (7692.4-7694.9) | 1575.8 (1575.2-1577.2) | **4.88x** | 100 | 99 | exact |
| 64 | 7699.8 (7699.6-7701.9) | 1584.1 (1581.9-1588.0) | **4.86x** | 100 | 98 | exact |

### Rate = 150 MB/s (Deck eMMC-ish)

| B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|
| 16 | 5132.0 (5131.1-5132.3) | 1192.7 (1190.6-1200.8) | **4.30x** | 150 | 131 | exact |
| 32 | 5134.2 (5133.7-5136.8) | 1199.1 (1196.6-1202.0) | **4.28x** | 150 | 130 | exact |
| 64 | 5140.7 (5140.3-5141.4) | 1204.4 (1199.5-1207.3) | **4.27x** | 149 | 130 | exact |

### Rate = 400 MB/s (fast USB / eMMC upper)

| B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|
| 16 | 1933.0 (1932.7-1940.8) | 863.7 (862.8-864.0) | **2.24x** | 397 | 181 | exact |
| 32 | 1936.2 (1935.8-1937.7) | 866.2 (864.4-869.5) | **2.24x** | 397 | 180 | exact |
| 64 | 1941.9 (1941.0-1943.9) | 868.1 (860.2-872.8) | **2.24x** | 395 | 180 | exact |

### Uncapped (R = 0, same harness baseline)

| B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|
| 16 | 296.9 (279.2-348.3) | 141.6 (132.3-149.6) | **2.10x** | 2587 | 1102 | exact |
| 32 | 298.3 (287.4-311.5) | 159.0 (152.5-209.9) | **1.88x** | 2574 | 981 | exact |
| 64 | 386.9 (332.1-402.1) | 209.5 (203.8-243.5) | **1.85x** | 1985 | 745 | exact |

Under Deck/SBC caps the cold lead is much larger than the prior ~1.2x to 2.1x virtual-disk memmap series: at 100 MB/s pair is ~4.86x to 4.88x across B = 16 / 32 / 64 (about 2.3x to 4.0x the uncapped memmap cold leads), at 150 MB/s ~4.27x to 4.30x, and at 400 MB/s still a clear **2.24x**. The same-harness uncapped baseline lands at 1.85x to 2.10x — consistent with the prior ~2.5 GB/s neighborhood on this box (N = 2M here vs 24M memmap earlier; GEMM cost is smaller so uncapped ratios sit a bit higher than the 24M B = 64 ~1.2x to 1.38x). At capped rates wall time is almost flat in B (I/O dominates), so the ask concurrent width does not eat the lead the way it does on the uncapped / compute-bound path. Memory remains 4.92x smaller (already known); the speed question under rate-cap is what this run answers.

## Verdict

**Viable.** On a Deck/SBC-class rate-capped cold ask at 2,000,000 rows (5 ms latency, 2 in flight, 8,192-row chunks, 8 workers), pair-code unpack is **4.86x to 4.88x** faster than int8 at 100 MB/s, **4.27x to 4.30x** at 150 MB/s, and **2.24x** at 400 MB/s for B = 16 / 32 / 64, with exact top-10 match every batch. That is substantially larger than the ~1.2x to 2.1x uncapped virtual-disk cold leads cited from earlier ask notes. Same-harness uncapped baseline is 1.85x to 2.10x. Memory is also 4.92x smaller. Bounded to this synthetic / modeled-device setup; real Deck/SBC media may differ, but the modeled caps clear the 1.5x bar with margin.

Caveats: synthetic embeddings; modeled device (latency + rate semaphore), not a physical SD/eMMC; N = 2M (established rate-cap size; not 24M); numpy / interpreter overhead; shared 8 vCPU box; one seed, five cold repeats per cell; MemAvailable during timed cells ~6.0 to 6.2 GB.

## Pair files this wave

- No pair file for the 14:05 wave reached this run (Chief both pairs no gap for 50+ waves; Trit Pack / Winsor / Side-Commit / Handshake correctly silent; roadmap blobs still JuniorHome ROADMAP 4efb666 / JuniorLLM BETA_TO_OS bac6923). JuniorLLM's tip after the 10:55 push was [583b07e](https://github.com/cloudcover95/JuniorLLM/commit/583b07e); this note adds one new file under `junior_osai/vault/findings/` only. The `ask` side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Rate-cap alone clears 1.5x with margin at B = 16 / 32 / 64. Does stacking a Deck/SBC rate cap (100 or 150 MB/s) with the memory-cap regime (int8 past MemAvailable) widen the cold lead further, or does the GEMM/unpack cost on the smaller resident working set change the crossover vs the I/O-bound ~4.3x to 4.9x seen here?

close: 0e8055e29af3094d (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-04_1055)
