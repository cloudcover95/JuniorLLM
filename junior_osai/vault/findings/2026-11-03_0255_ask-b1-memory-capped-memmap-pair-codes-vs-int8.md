---
local: true
date: 2026-11-03
wave: "2026-11-03 02:05 (bench 02:55, ran 03:03 to 03:22 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` scoring a B=1 interactive query against a memory-capped memmap index)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, ask, memmap, memory-cap, gemm, bench, b1]
---

# juniorctl ask, B = 1 interactive, memory-capped memmap: pair codes still beat int8 by about 4.5x

## Question

The 18:55 note ([[2026-11-02_1855_ask-b8-memory-capped-memmap-pair-codes-vs-int8]]) showed B = 8 ask viable at 3.77x to 3.83x warm on a 24M out-of-RAM memmap. Its next question is the one measured here. Does B = 1 interactive ask against the same 24M out-of-RAM memmap index still give pair codes ≥1.5x vs int8, or does lost GEMM amortization drop the lead below 1.5x (as resident B = 1 did)?

## Method

Synthetic 24,000,000-row x 384-dim index built with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), streamed to two flat files under the scratch `data/` dir and fsynced. Int8 file is 9.2 GB; pair file is 1.87 GB (4.92x smaller). MemAvailable after each build was 6.1 to 6.6 GB, so int8 cannot stay resident while pair can. Files were then dropped from Python and reopened as read-only `np.memmap`. Each B = 1 interactive query is scored per 8,192-row chunk (one unpack or cast per chunk, then an N_chunk x 384 by 384 x 1 GEMM), top-10 per chunk, merged. OpenBLAS pinned to 1 thread, 8 worker threads. Two seeds; the second runs pair before int8. 12 timed batches per warm cell after one priming pass; 5 timed batches per cold cell with `posix_fadvise(DONTNEED)` and remap before every batch. The first two batches of every cell are checked query by query against an exact single-threaded int8 memmap reference. numpy 2.5.4, 8 vCPU shared box, 16 GB RAM, python3 and numpy only, no network, nothing bound. A 131,072-row smoke run passed first (top-10 exact). Script (`bench.py`, scratch folder) imports `build`, `V10`, `D`, `PW`, `TOP`, `NC` from the 2026-10-30 14:55 scratch script; CLI B set to 1. Commands, data files deleted afterwards:

```
python3 bench.py 131072 202611030255 4 1 warm-int8,warm-pair,cold-int8,cold-pair          # smoke
python3 bench.py 24000000 202611030255 12 1 warm-int8,warm-pair
python3 bench.py 24000000 202611030256 12 1 warm-pair,warm-int8                           # second seed, reversed
python3 bench.py 24000000 202611030255 5 1 cold-pair,cold-int8
```

## Results

Median ms per B = 1 query, p10 to p90 in brackets. Top-10 matched the int8 reference exactly (100.00%) in every checked batch. "pair vs int8" is int8 time divided by pair time.

| cell | file bytes | seed ...0255 ms | seed ...0256 ms (reversed) | ms per query | pair vs int8 |
|---|---|---|---|---|---|
| warm int8 | 9,216,000,000 | 3715.7 (3549.6-3795.6) | 3543.1 (3427.2-3760.6) | 3543 to 3716 | 1.00x |
| warm pair | 1,872,000,000 | 817.1 (805.2-854.9) | 762.4 (744.7-789.3) | 762 to 817 | **4.55x / 4.65x** |
| cold int8 | 9,216,000,000 | 3484.6 (3426.9-3591.9) | — | 3485 | 1.00x |
| cold pair | 1,872,000,000 | 1042.3 (1029.3-1100.9) | — | 1042 | **3.34x** |

Warm, int8 is stuck near its cold cost (3.5 to 3.7 s) because 9.2 GB does not fit in about 6.1 to 6.6 GB MemAvailable, while pair stays around 0.76 to 0.82 s (most of its 1.87 GB served from cache). That is a 4.55x to 4.65x latency win for pair, well above the 1.5x bar. Cold (forced eviction every batch) still favors pair 3.34x: the disk delivers int8 at about 2.64 GB/s (3.48 s for 9.2 GB), but pair reads 4.92x fewer bytes and finishes in 1.04 s.

Compared with the resident 10:55 B = 1 cells (pair best 0.74x to 0.88x of int8 on 1M to 4M rows), the regime flip is complete: the same interactive B = 1 ask that lost on a resident index wins by about 4.5x once the index is memory-capped. Lost GEMM amortization (B = 1 vs B = 8) does not drop the lead below 1.5x here; if anything the warm lead is larger than B = 8's 3.77x to 3.83x because pair's compute floor matters less when only one query is scored and int8 remains fully I/O-bound. Absolute pair time is also lower at B = 1 (762 to 817 ms) than at B = 8 (1079 to 1131 ms), as expected when the GEMM width shrinks from 8 to 1 while the memmap read stays the same size.

## Verdict

**Viable.** On a memory-capped memmap ask at 24,000,000 rows (int8 9.2 GB past MemAvailable, pair 1.87 GB fitting), a B = 1 interactive query on pair codes is 4.55x to 4.65x faster warm and 3.34x faster cold than int8, with exact top-10 match. Memory is also 4.92x smaller (already known). For `ask`, use int8 while the index fits in RAM (10:55 / 14:55), and switch to pair-code memmap once it does not; the latency bar clears cleanly in the out-of-RAM regime at both B = 8 and B = 1.

Caveats: synthetic embeddings; numpy temporary arrays and the interpreter add overhead a compiled kernel would not; shared 8 vCPU box; the virtual disk (about 2.6 GB/s cold) is faster than Deck or SBC targets, so the cold lead would be larger on a slower device; two warm seeds only, one cold seed.

## Pair files this wave

- No pair file for the 02:05 wave reached this run, and no new roadmap item is named; the roadmap blobs have had nothing uncovered since 2026-10-24 (about the 47th idle wave per prior ledger; this is the next idle after 18:05), and the ledger will confirm the Chief's call. JuniorLLM's tip after the 18:55 push was [640cd21](https://github.com/cloudcover95/JuniorLLM/commit/640cd21); this note adds one new file under `junior_osai/vault/findings/` only. The `ask` side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

With ask now viable under a memory-capped memmap at both B = 8 and B = 1, does a small concurrent ask (B = 2 / 4) on the same 24M out-of-RAM memmap sit between those leads, or is there a non-monotonic crossover once GEMM amortization and page-cache pressure interact? Alternate: does B = 16 / 32 on the same capped memmap keep the ≥1.5x bar, or does pair become compute-bound the way night B = 64 did (1.48x borderline)?

close: 150a13a4d5cd0eb5 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-02_1855)
