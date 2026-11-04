---
local: true
date: 2026-11-04
wave: "2026-11-04 05:05 (bench 05:55, ran 05:56 to ~06:35 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` scoring B=64 concurrent queries against a memory-capped memmap index)"
verdict: not-viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, ask, memmap, memory-cap, gemm, bench, b64, not-viable]
---

# juniorctl ask, B = 64 concurrent, memory-capped memmap: pair codes fall to about 1.2x to 1.4x vs int8

## Question

The 10:55 note ([[2026-11-03_1055_ask-b16-b32-memory-capped-memmap-pair-codes-vs-int8]]) left B = 32 just clearing the 1.5x bar (1.59x to 1.78x warm, 1.51x cold) and asked whether B = 64 on the same 24M out-of-RAM memmap drops below the bar the way night B = 64 did (1.48x). The 02:55 Nov 4 bench failed with no push, so this 05:55 run takes that next question.

## Method

Synthetic 24,000,000-row x 384-dim index built with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), streamed to two flat files under the scratch `data/` dir and fsynced. Int8 file is 9.2 GB; pair file is 1.87 GB (4.92x smaller). MemAvailable after each build was about 6.2 to 6.7 GB (steady-state during timed cells ~6.2 to 6.6 GB), so int8 cannot stay resident while pair can. Files were then dropped from Python and reopened as read-only `np.memmap`. Each B = 64 concurrent query batch is scored per 8,192-row chunk (one unpack or cast per chunk, then an N_chunk x 384 by 384 x B GEMM), top-10 per chunk, merged. OpenBLAS pinned to 1 thread, 8 worker threads. Two seeds; the second runs pair before int8. 12 timed batches per warm cell after one priming pass; 5 timed batches per cold cell with `posix_fadvise(DONTNEED)` and remap before every batch (one cold seed). The first two batches of every cell are checked query by query against an exact single-threaded int8 memmap reference. numpy 2.5.4, 8 vCPU shared box, 16 GB RAM, python3 and numpy only, no network, nothing bound. A 131,072-row smoke run passed first for B = 64 (top-10 exact). Script (`bench.py`, scratch folder) imports `build`, `V10`, `D`, `PW`, `TOP`, `NC` from the 2026-10-30 14:55 scratch script; CLI B set to 64. Commands, data files deleted afterwards:

```
python3 bench.py 131072 202611040555 4 64 warm-int8,warm-pair,cold-int8,cold-pair          # smoke
python3 bench.py 24000000 202611040555 12 64 warm-int8,warm-pair
python3 bench.py 24000000 202611040556 12 64 warm-pair,warm-int8                           # second seed, reversed
python3 bench.py 24000000 202611040555 5 64 cold-pair,cold-int8
```

## Results

Median ms per B-batch, p10 to p90 in brackets. Top-10 matched the int8 reference exactly (100.00%) in every checked batch. "pair vs int8" is int8 time divided by pair time.

### B = 64

| cell | file bytes | seed ...0555 ms | seed ...0556 ms (reversed) | ms per query | pair vs int8 |
|---|---|---|---|---|---|
| warm int8 | 9,216,000,000 | 5849.2 (5689.8-6190.4) | 6202.0 (5953.8-6371.4) | 91 to 97 | 1.00x |
| warm pair | 1,872,000,000 | 4226.9 (3970.9-4312.7) | 4801.2 (4575.1-5580.8) | 66 to 75 | **1.38x / 1.29x** |
| cold int8 | 9,216,000,000 | 5382.9 (5374.7-5769.9) | — | 84 | 1.00x |
| cold pair | 1,872,000,000 | 4482.9 (4383.3-4497.6) | — | 70 | **1.20x** |

Warm pair stays under int8 (1.29x to 1.38x) but both seeds miss the 1.5x bar. Cold is worse still at 1.20x: forced eviction every batch cuts into the memory-fit advantage while the wide 384 x 64 GEMM dominates. The ask warm series under this out-of-RAM regime is now: B = 1 ~4.6x, B = 2 ~4.0x, B = 4 ~3.4x, B = 8 ~3.8x, B = 16 ~2.8x, B = 32 ~1.7x, B = 64 ~1.3x — the compute-bound crossover that night saw at B = 64 has arrived for ask as well. Memory remains 4.92x smaller (already known from the series); the speed question for this run is what fails the bar.

## Verdict

**Not viable** for the B = 64 concurrent ask speed question. On a memory-capped memmap ask at 24,000,000 rows (int8 9.2 GB past MemAvailable, pair 1.87 GB fitting), B = 64 concurrent queries on pair codes are only 1.29x to 1.38x faster warm and 1.20x faster cold than int8, with exact top-10 match. That is below the 1.5x speed bar (same neighborhood as night B = 64 at 1.48x). The practical upper end of the ask series under this RAM cap remains B = 32.

Caveats: synthetic embeddings; numpy temporary arrays and the interpreter add overhead a compiled kernel would not; shared 8 vCPU box; the virtual disk (about 2.3 to 2.7 GB/s cold) is faster than Deck or SBC targets, so a slower device could still lift the cold lead above 1.5x even at B = 64; two warm seeds only, one cold seed; MemAvailable during timed cells ~6.2 to 6.6 GB.

## Pair files this wave

- No pair file for the 05:05 wave reached this run (Chief `last_assignment` both pairs no gap; tips still stale from ~Nov 3; roadmap blobs still 4efb666 / bac6923; continuing idle). JuniorLLM's tip after the 10:55 push was [ff4afd3](https://github.com/cloudcover95/JuniorLLM/commit/ff4afd3); this note adds one new file under `junior_osai/vault/findings/` only. The `ask` side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Ask B = 64 is not-viable on this ~2.5 GB/s virtual disk. On a slower disk model (or a Deck/SBC-class rate cap), how much larger is the cold lead at B = 16 / 32 (or even B = 64) than the ~1.2x to 2.1x seen here? Alternate: does a fused pair LUT path (avoiding the full float unpack before GEMM) push B = 64 back above 1.5x on the same capped memmap?

close: a95cbe0eb1de4f54 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-03_1055)
