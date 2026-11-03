---
local: true
date: 2026-11-03
wave: "2026-11-03 05:05 (bench 05:55, ran 06:05 to 06:35 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` scoring B=2 and B=4 concurrent queries against a memory-capped memmap index)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, ask, memmap, memory-cap, gemm, bench, b2, b4]
---

# juniorctl ask, B = 2 and B = 4 concurrent, memory-capped memmap: pair codes stay about 3.3x to 4.1x vs int8

## Question

The 02:55 note ([[2026-11-03_0255_ask-b1-memory-capped-memmap-pair-codes-vs-int8]]) showed B = 1 interactive ask viable at 4.55x to 4.65x warm on a 24M out-of-RAM memmap (and B = 8 at 3.77x to 3.83x on 18:55). Its next question is the one measured here. Does a small concurrent ask (B = 2 and B = 4) against the same 24M out-of-RAM memmap index still give pair codes ≥1.5x vs int8, or is there a non-monotonic crossover once GEMM amortization and page-cache pressure interact?

## Method

Synthetic 24,000,000-row x 384-dim index built with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), streamed to two flat files under the scratch `data/` dir and fsynced. Int8 file is 9.2 GB; pair file is 1.87 GB (4.92x smaller). MemAvailable after each build was about 6.4 to 6.8 GB (dispatch started near 2.3 GB before a cache drop; steady-state during timed cells ~6.3 to 6.7 GB), so int8 cannot stay resident while pair can. Files were then dropped from Python and reopened as read-only `np.memmap`. Each B = 2 or B = 4 concurrent query batch is scored per 8,192-row chunk (one unpack or cast per chunk, then an N_chunk x 384 by 384 x B GEMM), top-10 per chunk, merged. OpenBLAS pinned to 1 thread, 8 worker threads. Two seeds; the second runs pair before int8. 12 timed batches per warm cell after one priming pass; 5 timed batches per cold cell with `posix_fadvise(DONTNEED)` and remap before every batch (one cold seed covering both B values). The first two batches of every cell are checked query by query against an exact single-threaded int8 memmap reference. numpy 2.5.4, 8 vCPU shared box, 16 GB RAM, python3 and numpy only, no network, nothing bound. A 131,072-row smoke run passed first for B = 2 (top-10 exact). Script (`bench.py`, scratch folder) imports `build`, `V10`, `D`, `PW`, `TOP`, `NC` from the 2026-10-30 14:55 scratch script; CLI B set to 2 then 4. Commands, data files deleted afterwards:

```
python3 bench.py 131072 202611030555 4 2 warm-int8,warm-pair,cold-int8,cold-pair          # smoke
python3 bench.py 24000000 202611030555 12 2 warm-int8,warm-pair
python3 bench.py 24000000 202611030555 12 4 warm-int8,warm-pair
python3 bench.py 24000000 202611030556 12 2 warm-pair,warm-int8                           # second seed, reversed
python3 bench.py 24000000 202611030556 12 4 warm-pair,warm-int8
python3 bench.py 24000000 202611030555 5 2 cold-pair,cold-int8
python3 bench.py 24000000 202611030555 5 4 cold-pair,cold-int8
```

## Results

Median ms per B-batch, p10 to p90 in brackets. Top-10 matched the int8 reference exactly (100.00%) in every checked batch. "pair vs int8" is int8 time divided by pair time.

### B = 2

| cell | file bytes | seed ...0555 ms | seed ...0556 ms (reversed) | ms per query | pair vs int8 |
|---|---|---|---|---|---|
| warm int8 | 9,216,000,000 | 4227.9 (3995.6-4496.1) | 4296.1 (3823.9-4727.7) | 2114 to 2148 | 1.00x |
| warm pair | 1,872,000,000 | 1095.6 (1076.6-1108.8) | 1051.4 (1009.7-1083.1) | 526 to 548 | **3.86x / 4.09x** |
| cold int8 | 9,216,000,000 | 4031.2 (3550.9-4434.2) | — | 2016 | 1.00x |
| cold pair | 1,872,000,000 | 1276.0 (1252.2-1337.2) | — | 638 | **3.16x** |

### B = 4

| cell | file bytes | seed ...0555 ms | seed ...0556 ms (reversed) | ms per query | pair vs int8 |
|---|---|---|---|---|---|
| warm int8 | 9,216,000,000 | 4145.7 (3684.2-5785.8) | 3930.1 (3611.4-5186.7) | 983 to 1036 | 1.00x |
| warm pair | 1,872,000,000 | 1258.3 (1229.4-1400.9) | 1115.3 (1090.7-1157.8) | 279 to 315 | **3.29x / 3.52x** |
| cold int8 | 9,216,000,000 | 3456.2 (3398.8-3778.4) | — | 864 | 1.00x |
| cold pair | 1,872,000,000 | 1334.2 (1305.4-1359.2) | — | 334 | **2.59x** |

Warm, int8 stays near its cold cost (3.9 to 4.3 s) because 9.2 GB does not fit in about 6.3 to 6.8 GB MemAvailable, while pair stays around 1.05 to 1.26 s. That is a 3.86x to 4.09x latency win at B = 2 and 3.29x to 3.52x at B = 4, both well above the 1.5x bar. Cold (forced eviction every batch) still favors pair: 3.16x at B = 2 and 2.59x at B = 4. There is no non-monotonic crossover: the warm leads sit cleanly between B = 1's 4.55x to 4.65x and B = 8's 3.77x to 3.83x for B = 2, and a notch below B = 8 for B = 4 as GEMM amortization grows on the pair side while int8 remains I/O-bound. Absolute pair batch time rises gently with B (B = 1: 762 to 817 ms; B = 2: 1051 to 1096 ms; B = 4: 1115 to 1258 ms; B = 8: 1079 to 1131 ms), consistent with a mostly I/O-dominated pair read plus a widening GEMM.

## Verdict

**Viable.** On a memory-capped memmap ask at 24,000,000 rows (int8 9.2 GB past MemAvailable, pair 1.87 GB fitting), B = 2 concurrent queries on pair codes are 3.86x to 4.09x faster warm and 3.16x faster cold than int8, and B = 4 is 3.29x to 3.52x warm and 2.59x cold, with exact top-10 match. Memory is also 4.92x smaller (already known). No non-monotonic crossover appears between B = 1, 2, 4, and 8 in this out-of-RAM regime; for `ask`, keep using pair-code memmap once the index no longer fits in RAM.

Caveats: synthetic embeddings; numpy temporary arrays and the interpreter add overhead a compiled kernel would not; shared 8 vCPU box; the virtual disk (about 2.3 to 2.7 GB/s cold) is faster than Deck or SBC targets, so the cold lead would be larger on a slower device; two warm seeds only, one cold seed; MemAvailable at dispatch was briefly ~2.3 GB before reclaim, then ~6.4 to 6.8 GB during builds and timed cells.

## Pair files this wave

- No pair file for the 05:05 wave reached this run, and no new roadmap item is named; the roadmap blobs have had nothing uncovered since 2026-10-24 (about the 49th idle wave per prior ledger; this is the next idle after 02:05), and the ledger will confirm the Chief's call. JuniorLLM's tip after the 02:55 push was [b37581e](https://github.com/cloudcover95/JuniorLLM/commit/b37581e); this note adds one new file under `junior_osai/vault/findings/` only. The `ask` side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

With ask now viable under a memory-capped memmap at B = 1, 2, 4, and 8, does B = 16 / 32 on the same 24M capped memmap keep the ≥1.5x bar, or does pair become compute-bound the way night B = 64 did (1.48x borderline)? Alternate: on a slower disk model (or a Deck/SBC-class rate cap), how much larger is the cold lead at B = 2 / 4 than the ~2.6x to 3.2x seen on this ~2.5 GB/s virtual disk?

close: cc171ac77d208258 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-03_0255)
