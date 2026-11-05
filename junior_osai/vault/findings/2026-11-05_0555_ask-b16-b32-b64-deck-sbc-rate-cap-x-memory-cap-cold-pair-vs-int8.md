---
local: true
date: 2026-11-05
wave: "2026-11-05 05:05 (bench 05:55, ran ~05:57–06:47 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` cold B=16/32/64 Deck/SBC rate-cap × memory-cap stack pair vs int8)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, ask, cold, rate-cap, memory-cap, deck, sbc, bench, b16, b32, b64, viable, stack]
---

# juniorctl ask, cold B = 16 / 32 / 64 under Deck/SBC rate caps with int8 past MemAvailable: pair codes ~4.3x to 4.9x vs int8

## Question

The 14:55 note ([[2026-11-04_1455_ask-b16-b32-b64-deck-sbc-rate-capped-cold-pair-vs-int8]]) cleared rate-cap alone at N = 2M (~4.86x to 4.88x at 100 MB/s, ~4.27x to 4.30x at 150 MB/s) and asked whether stacking a Deck/SBC rate cap with the memory-cap regime (int8 past MemAvailable) widens the cold lead further, or whether GEMM/unpack on the smaller resident working set changes the crossover. This 05:55 run answers that stack question (ternary compute / ask path).

## Method

Synthetic 20,000,000-row x 384-dim index built with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), streamed to two flat files under the scratch dir and fsynced. Int8 file is 7.68 GB; pair file is 1.56 GB (4.92x smaller). At start MemAvailable was ~6.3 GB, so int8 is past MemAvailable (`int8_past_memavail True` every timed cell) while pair may still fit. Same device model as the 14:55 rate benches: fixed 5 ms latency per request, 2 requests in flight (semaphore), bytes then reserved one-at-a-time at R MB/s. Chunk size 8,192 rows; 8 worker threads; OpenBLAS / OMP pinned to 1. Scoring per chunk is int8 cast→float32 GEMM or pair `np.take(V10,…)` unpack→float32 GEMM, then chunked top-10 merge. Cold only: `posix_fadvise(DONTNEED)` before every timed batch. Batches B ∈ {16, 32, 64} slice the 64-query build matrix (rotating offset per repeat). ≥5 timed cold batches per cell; format order reversed on odd repeats. Every timed batch checked against an uncapped single-thread int8 reference (exact set equality). Rates: 100 MB/s (SBC SD / slow Deck SD) and 150 MB/s (Deck eMMC-ish). numpy 2.5.4, 8 vCPU shared box, python3 and numpy only, no network, nothing bound except the local files. A 131,072-row smoke at 100 MB/s / B = 16,32,64 passed first (top-10 exact, ~4.47x to 4.58x). Script (`bench.py`, scratch `/workspace/bench0555_1105/`) imports `build`, `V10`, `D`, `PW`, `TOP`, `Dev` from the 2026-10-30 14:55 scratch. Commands, data bins deleted afterwards:

```
python3 bench.py 131072 202611050555 1 100 5 8 2 8192 16,32,64
python3 bench.py 20000000 202611050555 5 100,150 5 8 2 8192 16,32,64
```

## Results

Median ms per B-batch, p10 to p90 in brackets. Top-10 matched the uncapped single-thread int8 reference exactly in every timed batch. "pair vs int8" is int8 median / pair median. Observed read rates are derived from file bytes / median wall time. MemAvail is MB MemAvailable after the cell; int8_past is whether int8 bytes exceed MemAvailable at that cell.

### Rate = 100 MB/s (SBC SD / slow Deck SD), int8 past MemAvailable

| B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 | MemAvail | int8_past |
|---|---|---|---|---|---|---|---|---|
| 16 | 76845.5 (76844.1-76854.0) | 15658.9 (15652.3-15674.4) | **4.91x** | 100 | 100 | exact | 6006 | True |
| 32 | 76875.1 (76874.2-76877.8) | 15682.2 (15679.0-15691.7) | **4.90x** | 100 | 99 | exact | 6640 | True |
| 64 | 76943.0 (76933.3-76947.1) | 15739.5 (15733.6-15753.7) | **4.89x** | 100 | 99 | exact | 6597 | True |

### Rate = 150 MB/s (Deck eMMC-ish), int8 past MemAvailable

| B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 | MemAvail | int8_past |
|---|---|---|---|---|---|---|---|---|
| 16 | 51242.5 (51240.5-51249.2) | 11843.9 (11806.8-11856.5) | **4.33x** | 150 | 132 | exact | 5924 | True |
| 32 | 51274.5 (51268.6-51277.6) | 11892.3 (11875.4-11901.8) | **4.31x** | 150 | 131 | exact | 6586 | True |
| 64 | 51336.9 (51334.4-51343.7) | 11960.1 (11951.6-11965.1) | **4.29x** | 150 | 130 | exact | 6606 | True |

Stacking memory-cap on the rate-cap does **not** widen the lead vs the N = 2M rate-cap-only note: at 100 MB/s the stack is ~4.89x to 4.91x (vs ~4.86x to 4.88x at N = 2M), and at 150 MB/s ~4.29x to 4.33x (vs ~4.27x to 4.30x). Ratios stay flat in B (I/O still dominates). Memory remains 4.92x smaller; the int8 file is ~7.68 GB against ~6.0 to 6.6 GB MemAvailable during cells, so the memory-cap flag is live, but the extra pressure does not move the crossover off the I/O-bound ~4.3x to 4.9x band.

## Verdict

**Viable.** On a Deck/SBC-class rate-capped cold ask stacked with int8 past MemAvailable at 20,000,000 rows (5 ms latency, 2 in flight, 8,192-row chunks, 8 workers), pair-code unpack is **4.89x to 4.91x** faster than int8 at 100 MB/s and **4.29x to 4.33x** at 150 MB/s for B = 16 / 32 / 64, with exact top-10 match every batch. The stack does not beat rate-cap alone by a meaningful margin; it holds the same I/O-bound band. Memory is also 4.92x smaller. Bounded to this synthetic / modeled-device setup; real Deck/SBC media and real page-cache pressure may differ, but the modeled stack clears the 1.5x bar with margin.

Caveats: synthetic embeddings; modeled device (latency + rate semaphore), not a physical SD/eMMC; N = 20M (chosen so int8 exceeds MemAvailable); numpy / interpreter overhead; shared 8 vCPU box; one seed, five cold repeats per cell; MemAvailable during timed cells ~5.9 to 6.6 GB; the 05:55 automation initially failed to vault-push, so this note is a catch-up publish of the completed local residue.

## Pair files this wave

- No pair file for the 05:05 wave reached this run (Chief both pairs no gap; Trit Pack / Winsor / Side-Commit / Handshake correctly silent; roadmap blobs still JuniorHome ROADMAP 4efb666 / JuniorLLM BETA_TO_OS bac6923). JuniorLLM tip before this push was [9c7cad2](https://github.com/cloudcover95/JuniorLLM/commit/9c7cad2) (rails noforce offline stdlib guard overnight, no vault findings). The `ask` side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Rate-cap alone and rate-cap × memory-cap both sit in the same ~4.3x to 4.9x I/O-bound band at B = 16 / 32 / 64. Does a real Deck/SBC media path (physical eMMC/SD, not the latency+rate semaphore) keep that lead, or does device queueing / readahead collapse it toward the uncapped ~1.8x to 2.1x neighborhood?

close: 53ee2ff319c95e3d (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-04_1455)
