---
local: true
date: 2026-11-02
wave: "2026-11-02 02:05 (bench 02:55, ran 03:00 to 03:11)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`night` sweeping the packed ask index)"
verdict: not-viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, night, ask-index, end-to-end, real-disk, page-cache, bench]
---

# juniorctl night end to end on the real disk at 8,000,000 rows: pair codes vs int8, cold and warm

## Question

The 05:55 note ([[2026-11-01_0555_night-float32-and-float16-pair-table-decode-vs-int8-unpack]]) settled the in-memory compute floor (pair unpack + GEMM about 100 ms per million rows, int8 10% to 15% faster) and asked whether the pair-code lead holds once the index outgrows page cache. Earlier `night` leads (about 2x) came from a modeled 1.6 GB/s device. On this box's real disk, with no device model, at 8,000,000 rows (int8 file 3.07 GB, pair file 624 MB, about 4.9x smaller), is a full `night` sweep over pair codes at least 1.5x faster than over int8, cold or warm?

## Method

Synthetic 8,000,000-row x 384-dim index built with the 2026-10-30 14:55 encoders unchanged (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), written to two flat files and fsynced. 64 queries, 8 threads; each thread `os.pread`s one 8,192-row chunk from the file and scores it (int8: cast + `Q @ X.T`; pair: `np.take` on the 59,049 x 10 int8 pair table, slice to 384, cast, `Q @ X.T`), OpenBLAS pinned to 1 thread, top-10 per query merged across 977 chunks, every sweep checked against a cold int8 reference sweep.

- `cold`: `posix_fadvise(DONTNEED)` on both files before every timed sweep, so every byte comes off the disk.
- `warm`: one untimed priming sweep, then back-to-back sweeps; the kernel keeps whatever it can.

5 timed sweeps per cell, two seeds with cells reordered (seed 202611020255 runs cold int8 and cold pair twice to expose drift). numpy 2.5.4, 8 vCPU shared box, 16 GB RAM; MemAvailable was about 2.5 GB when the run started but rose to about 6.3 GB once the 3.7 GB of files were built, so the int8 file still fit in page cache when warm. python3 and numpy only, no network, nothing bound. A 262,144-row smoke run came first (`python3 bench.py 262144 202611020255 3 cold-int8,cold-pair,warm-int8,warm-pair`, top-10 exact in every cell). The script is not committed (about 45 lines, imports `build`, `V10`, `D`, `PW`, `TOP` from the 2026-10-30 14:55 scratch script). Commands, run in a scratch folder, data files deleted afterwards:

```
python3 bench.py 8000000 202611020255 5 cold-int8,cold-pair,warm-int8,warm-pair,cold-pair,cold-int8   # N, seed, repeats, cells
python3 bench.py 8000000 202611020256 5 warm-pair,cold-pair,warm-int8,cold-int8
```

## Results

Median of 5 sweeps over 8,000,000 rows in ms, spread (max minus min) in brackets. Where a cell ran twice both medians are shown. Top-10 matched the int8 reference exactly (100.00%) in every cell.

| cell | file bytes | seed ...255 ms | seed ...256 ms | ms per 1M rows | file read rate | pair vs int8 (same seed) |
|---|---|---|---|---|---|---|
| cold int8 | 3,072,000,000 | 1,342 (95), 1,199 (190) | 1,227 (96) | 157 to 176 | 2.3 to 2.6 GB/s | 1.00x |
| cold pair | 624,000,000 | 933 (149), 819 (40) | 977 (88) | 107 to 128 | 0.64 to 0.76 GB/s | 1.45x / 1.26x |
| warm int8 | 3,072,000,000 | 785 (68) | 789 (153) | 103 | 3.9 GB/s | 1.00x |
| warm pair | 624,000,000 | 771 (71) | 869 (121) | 101 to 114 | 0.72 to 0.81 GB/s | 1.02x / 0.91x |

Cold, the box's disk delivers int8 at 2.3 to 2.6 GB/s, so reading 3.07 GB costs only about 400 to 550 ms over the compute floor. The pair path reads 4.9x fewer bytes but is compute-bound: its cold sweep (819 to 977 ms) sits only 50 to 200 ms above its own warm sweep, and its warm sweep is no faster than int8's. The lead is 1.45x on the first seed (average 1,270 vs 876 ms) and 1.26x on the second (1,227 vs 977 ms). Warm, both files stay in page cache (MemAvailable about 6 GB), and the two paths are level within noise (1.02x and 0.91x).

## Verdict

**Not viable.** On this box's real disk at 8,000,000 rows the pair-code `night` sweep is 1.26x to 1.45x faster than int8 when both are cold, short of the 1.5x bar, and level (0.91x to 1.02x) once page cache holds both. The about 2x leads in earlier notes depend on a device slower than about 1.6 GB/s; on a fast NVMe-class disk the 4.9x read saving is mostly hidden behind the numpy unpack floor. Memory still favors pairs (624 MB vs 3.07 GB on disk and in cache, 4.9x), but that was already recorded and is not a sweep-speed win here.

Caveats: synthetic embeddings; two seeds; shared 8 vCPU box, so differences under about 10% are inside the noise; the int8 file never fell out of cache because MemAvailable rose to about 6.3 GB after the build, so a true "int8 does not fit, pair does" case was not reached; the disk is a virtual block device whose speed may not match the Deck or SBC targets.

## Pair files this wave

- No pair file for the 02:05 wave reached this run, and no new roadmap item is named; the roadmap blobs have had nothing uncovered since 2026-10-24, and the ledger will confirm the Chief's call. JuniorLLM's tip at clone time was [a85f4b4](https://github.com/cloudcover95/JuniorLLM/commit/a85f4b4) (cloudcover95, 02:53 MT, `rails/order`), which this note does not touch; this note adds one new file under `junior_osai/vault/findings/` only. The 10:55, 14:55 and 18:55 bench runs on 2026-11-01 left no note, so the previous note is still the 05:55 one. The ask/night side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

At what index size does int8 actually fall out of page cache on this box while pair codes stay resident: at 24,000,000 rows (int8 9.2 GB, pair 1.9 GB, above the about 6.3 GB MemAvailable seen here), does the warm pair-code `night` sweep beat warm int8 by 1.5x or more on the real disk?

close: 2b0ce34d5b9eb415 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-01_0555)
