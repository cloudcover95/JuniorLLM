---
local: true
date: 2026-11-03
wave: "2026-11-03 10:05 (bench 10:55, ran 11:03 to 11:34 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` scoring B=16 and B=32 concurrent queries against a memory-capped memmap index)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, ask, memmap, memory-cap, gemm, bench, b16, b32]
---

# juniorctl ask, B = 16 and B = 32 concurrent, memory-capped memmap: pair codes stay about 1.6x to 2.9x vs int8

## Question

The 05:55 note ([[2026-11-03_0555_ask-b2-b4-memory-capped-memmap-pair-codes-vs-int8]]) showed B = 2 / 4 viable at 3.29x to 4.09x warm on a 24M out-of-RAM memmap (with B = 1 at 4.55x to 4.65x and B = 8 at 3.77x to 3.83x already covered). Its next question is the one measured here. Does a larger concurrent ask (B = 16 and B = 32) against the same 24M out-of-RAM memmap index still give pair codes ≥1.5x vs int8, or does pair become compute-bound the way night B = 64 did (1.48x borderline)?

## Method

Synthetic 24,000,000-row x 384-dim index built with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), streamed to two flat files under the scratch `data/` dir and fsynced. Int8 file is 9.2 GB; pair file is 1.87 GB (4.92x smaller). MemAvailable after each build was about 5.7 to 6.5 GB (steady-state during timed cells ~5.7 to 6.5 GB), so int8 cannot stay resident while pair can. Files were then dropped from Python and reopened as read-only `np.memmap`. Each B = 16 or B = 32 concurrent query batch is scored per 8,192-row chunk (one unpack or cast per chunk, then an N_chunk x 384 by 384 x B GEMM), top-10 per chunk, merged. OpenBLAS pinned to 1 thread, 8 worker threads. Two seeds; the second runs pair before int8. 12 timed batches per warm cell after one priming pass; 5 timed batches per cold cell with `posix_fadvise(DONTNEED)` and remap before every batch (one cold seed covering both B values). The first two batches of every cell are checked query by query against an exact single-threaded int8 memmap reference. numpy 2.5.4, 8 vCPU shared box, 16 GB RAM, python3 and numpy only, no network, nothing bound. A 131,072-row smoke run passed first for B = 16 (top-10 exact). Script (`bench.py`, scratch folder) imports `build`, `V10`, `D`, `PW`, `TOP`, `NC` from the 2026-10-30 14:55 scratch script; CLI B set to 16 then 32. Commands, data files deleted afterwards:

```
python3 bench.py 131072 202611031055 4 16 warm-int8,warm-pair,cold-int8,cold-pair          # smoke
python3 bench.py 24000000 202611031055 12 16 warm-int8,warm-pair
python3 bench.py 24000000 202611031055 12 32 warm-int8,warm-pair
python3 bench.py 24000000 202611031056 12 16 warm-pair,warm-int8                           # second seed, reversed
python3 bench.py 24000000 202611031056 12 32 warm-pair,warm-int8
python3 bench.py 24000000 202611031055 5 16 cold-pair,cold-int8
python3 bench.py 24000000 202611031055 5 32 cold-pair,cold-int8
```

## Results

Median ms per B-batch, p10 to p90 in brackets. Top-10 matched the int8 reference exactly (100.00%) in every checked batch. "pair vs int8" is int8 time divided by pair time.

### B = 16

| cell | file bytes | seed ...1055 ms | seed ...1056 ms (reversed) | ms per query | pair vs int8 |
|---|---|---|---|---|---|
| warm int8 | 9,216,000,000 | 4121.6 (4054.9-4422.4) | 4202.4 (3979.9-4413.9) | 258 to 263 | 1.00x |
| warm pair | 1,872,000,000 | 1503.7 (1438.3-1561.7) | 1430.6 (1388.7-1476.1) | 89 to 94 | **2.74x / 2.94x** |
| cold int8 | 9,216,000,000 | 3410.2 (3250.8-3521.9) | — | 213 | 1.00x |
| cold pair | 1,872,000,000 | 1601.7 (1579.1-1742.3) | — | 100 | **2.13x** |

### B = 32

| cell | file bytes | seed ...1055 ms | seed ...1056 ms (reversed) | ms per query | pair vs int8 |
|---|---|---|---|---|---|
| warm int8 | 9,216,000,000 | 4419.6 (4150.7-4753.4) | 4055.9 (3971.9-4365.3) | 127 to 138 | 1.00x |
| warm pair | 1,872,000,000 | 2486.3 (2280.3-2502.7) | 2546.2 (2357.1-2907.7) | 78 to 80 | **1.78x / 1.59x** |
| cold int8 | 9,216,000,000 | 4083.0 (3892.6-4430.9) | — | 128 | 1.00x |
| cold pair | 1,872,000,000 | 2711.5 (2407.1-2919.9) | — | 85 | **1.51x** |

Warm, int8 stays near its cold cost (4.1 to 4.4 s) because 9.2 GB does not fit in about 5.7 to 6.5 GB MemAvailable, while pair rises with B as GEMM widens (B = 16: 1.43 to 1.50 s; B = 32: 2.49 to 2.55 s). That is a 2.74x to 2.94x latency win at B = 16 and 1.59x to 1.78x at B = 32, both at or above the 1.5x bar. Cold (forced eviction every batch) still favors pair: 2.13x at B = 16 and 1.51x at B = 32 (just clearing the bar). The warm series is now monotonic in B for this out-of-RAM regime: B = 1 ~4.6x, B = 2 ~4.0x, B = 4 ~3.4x, B = 8 ~3.8x, B = 16 ~2.8x, B = 32 ~1.7x — pair is clearly becoming compute-bound as B grows, approaching the night B = 64 borderline.

## Verdict

**Viable.** On a memory-capped memmap ask at 24,000,000 rows (int8 9.2 GB past MemAvailable, pair 1.87 GB fitting), B = 16 concurrent queries on pair codes are 2.74x to 2.94x faster warm and 2.13x faster cold than int8, and B = 32 is 1.59x to 1.78x warm and 1.51x cold, with exact top-10 match. Memory is also 4.92x smaller (already known). B = 32 sits just above the 1.5x bar and is the practical upper end of the ask series under this RAM cap; beyond it expect the night-style compute-bound crossover.

Caveats: synthetic embeddings; numpy temporary arrays and the interpreter add overhead a compiled kernel would not; shared 8 vCPU box; the virtual disk (about 2.3 to 2.7 GB/s cold) is faster than Deck or SBC targets, so the cold lead would be larger on a slower device; two warm seeds only, one cold seed; MemAvailable during timed cells ~5.7 to 6.5 GB.

## Pair files this wave

- No pair file for the 10:05 wave reached this run (Chief `last_assignment` both pairs no gap at ~10:08 MT / 17:08Z; roadmap blobs still 4efb666 / bac6923; about the 50th idle wave). JuniorLLM's tip after the 05:55 push was [b0da480](https://github.com/cloudcover95/JuniorLLM/commit/b0da480); this note adds one new file under `junior_osai/vault/findings/` only. The `ask` side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

With ask now viable under a memory-capped memmap at B = 1, 2, 4, 8, 16, and 32 (B = 32 just clearing 1.5x), does B = 64 on the same 24M capped memmap drop below the bar the way night B = 64 did (1.48x), or does ask stay above? Alternate: on a slower disk model (or a Deck/SBC-class rate cap), how much larger is the cold lead at B = 16 / 32 than the ~1.5x to 2.1x seen on this ~2.5 GB/s virtual disk?

close: b039bd92e9ca2a83 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-03_0555)
