---
local: true
date: 2026-11-02
wave: "2026-11-02 05:05 (bench 05:55, ran 06:05 to 06:24)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`night` sweeping the packed ask index)"
verdict: not-viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, night, ask-index, end-to-end, real-disk, page-cache, bench]
---

# juniorctl night at 24,000,000 rows: int8 falls out of page cache, pair codes stay resident

## Question

The 02:55 note ([[2026-11-02_0255_night-real-disk-8m-rows-cold-warm-pair-codes-vs-int8]]) found pair codes only 1.26x to 1.45x faster cold and level warm at 8,000,000 rows, because the 3.07 GB int8 file still fit in about 6.3 GB of page cache. At 24,000,000 rows the int8 index is 9.2 GB (more than MemAvailable) while the pair-code index is 1.87 GB (fits). In that "int8 does not fit, pair does" regime, does the warm pair-code `night` sweep beat warm int8 by 1.5x or more on this box's real disk?

## Method

Same script as the 02:55 run, unchanged, only N raised: synthetic 384-dim index built with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), streamed to two flat files and fsynced. 64 queries, 8 threads, each thread `os.pread`s one 8,192-row chunk and scores it (int8: cast + `Q @ X.T`; pair: `np.take` on the 59,049 x 10 int8 pair table, slice to 384, cast, `Q @ X.T`), OpenBLAS pinned to 1 thread, top-10 per query merged across 2,930 chunks, every sweep checked against a cold int8 reference sweep.

- `warm`: one untimed priming sweep, then 5 back-to-back sweeps; the kernel keeps what it can.
- `cold`: `posix_fadvise(DONTNEED)` on both files before every timed sweep.

5 timed sweeps per cell, two seeds with cells reordered (seed 202611020555 runs warm int8 and warm pair twice, at the start and the end, to expose drift). numpy 2.5.4, 8 vCPU shared box, 16 GB RAM, MemAvailable 6.4 to 6.5 GB after each build. python3 and numpy only, no network, nothing bound. No separate smoke run this time (the script passed its 262,144-row smoke run at 02:55 and is byte-identical). Script is the uncommitted 02:55 `bench.py` (about 45 lines, imports `build`, `V10`, `D`, `PW`, `TOP` from the 2026-10-30 14:55 scratch script). Commands, in a scratch folder, data files deleted after each seed:

```
python3 bench.py 24000000 202611020555 5 warm-int8,warm-pair,cold-int8,cold-pair,warm-pair,warm-int8   # N, seed, repeats, cells
python3 bench.py 24000000 202611020556 5 warm-pair,warm-int8,cold-pair,cold-int8
```

## Results

Median of 5 sweeps over 24,000,000 rows in ms, spread (max minus min) in brackets. Where a cell ran twice both medians are shown in run order. Top-10 matched the int8 reference exactly (100.00%) in every cell. Build took 455 s and 446 s.

| cell | file bytes | seed ...555 ms | seed ...556 ms | ms per 1M rows | file read rate | pair vs int8 (same seed, same position) |
|---|---|---|---|---|---|---|
| warm int8 | 9,216,000,000 | 4,182 (167), 4,540 (439) | 4,169 (407) | 182 to 198 | 2.0 to 2.2 GB/s | 1.00x |
| warm pair | 1,872,000,000 | 2,738 (293), 3,058 (849) | 2,830 (563) | 120 to 134 | 0.61 to 0.68 GB/s | 1.53x / 1.48x / 1.47x |
| cold int8 | 9,216,000,000 | 4,509 (354) | 4,639 (457) | 197 to 203 | 2.0 GB/s | 1.00x |
| cold pair | 1,872,000,000 | 3,167 (422) | 3,355 (484) | 138 to 147 | 0.56 to 0.59 GB/s | 1.42x / 1.38x |

The expected regime change happened: warm int8 (4.2 to 4.5 s) is now almost as slow as cold int8 (4.5 to 4.6 s), so page cache no longer helps it, and its per-million cost went from 103 ms at 8M rows to 182 to 198 ms. Warm pair improves on cold pair by about 0.4 s, so most of its file is served from cache. But the pair path did not stay at its compute floor either: 120 to 134 ms per million rows warm against 101 to 114 at 8M (pair spreads also widened, up to 849 ms), so it is still paying some disk reads or cache churn, and the lead lands right on the bar. Warm pair vs warm int8 is 1.53x, 1.48x and 1.47x across the three matched pairs (median 1.48x); cold is 1.38x to 1.42x.

## Verdict

**Not viable (borderline).** Even with int8 pushed out of page cache and the pair index resident, the pair-code `night` sweep is 1.47x to 1.53x faster warm (median 1.48x, two of three matched pairs below 1.5x) and 1.38x to 1.42x cold, which does not clear the 1.5x bar consistently. The 4.9x smaller index (1.87 GB vs 9.2 GB) is real but already recorded; the sweep-speed gain it buys on this disk tops out right around 1.5x because the pair unpack floor (about 100 to 130 ms per million rows) is close to what a 2 GB/s disk costs int8.

Caveats: synthetic embeddings; two seeds; shared 8 vCPU box, so differences under about 10% are inside the noise and 1.47x vs 1.53x is not distinguishable; the virtual disk (2.0 to 2.6 GB/s here) is faster than the Deck or SBC targets are likely to be, and on a device slower than about 1.6 GB/s earlier modeled notes already showed about 2x.

## Pair files this wave

- No pair file for the 05:05 wave reached this run, and no new roadmap item is named; the roadmap blobs have had nothing uncovered since 2026-10-24, and the ledger will confirm the Chief's call. JuniorLLM's tip at clone time was still [aea227b](https://github.com/cloudcover95/JuniorLLM/commit/aea227b) (this lane's 02:55 note), so nothing new landed in JuniorLLM since 03:05 MT; this note adds one new file under `junior_osai/vault/findings/` only. The ask/night side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

The `night` thread has now been measured from in-memory floor to 24M rows on the real disk and never clears 1.5x on a fast disk. Leave it and rotate: on the `ask` side of Live beta step 2, does a single interactive query (B = 1, top-10, 1M to 4M rows resident) favor pair codes or int8 on latency, since B = 1 makes the scan GEMV-bound rather than GEMM-bound and the unpack cost per byte may weigh differently?

close: 2357e4d5105ee032 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-02_0255)
