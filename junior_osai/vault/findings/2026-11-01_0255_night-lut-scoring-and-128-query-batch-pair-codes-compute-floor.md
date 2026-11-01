---
local: true
date: 2026-11-01
wave: "2026-11-01 02:05 (bench 02:55, ran 03:05 to 03:12)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`night` sweeping the packed ask index)"
verdict: not-viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, night, ask-index, lookup-table, batching, compute-floor, bench]
---

# juniorctl night compute floor: scoring straight from pair codes with lookup tables, and 128 vs 64 queries per chunk

## Question

The 18:55 note ([[2026-10-31_1855_night-reader-pool-past-scorers-second-seed-pair-codes-vs-int8]]) found that once 8 readers + 8 scorers hide storage, uint16 pair codes sit at about 260 ms of decode plus scoring for 2,000,000 rows x 64 queries. Can that compute floor be cut by 1.5x or more, either by scoring straight from the pair codes with per-query lookup tables (no float unpack, no GEMM) or by batching 128 queries per chunk instead of 64?

## Method

In-memory sweep (storage taken out, since the 18:55 layout already hides it) over a synthetic 1,048,576-row x 384-dim index, seed 202611010255, built with the 2026-10-30 14:55 encoders unchanged (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row). 8 scoring threads over 8,192-row chunks (128 chunks), OpenBLAS pinned to 1 thread, top-10 per query merged across chunks. Modes:

- `int8`: int8 rows to float32, `Q @ X.T`.
- `unpack`: current pair path, `np.take` on the 59,049 x 10 pair table, float32, `Q @ X.T`.
- `lut5`: per batch, a (78, 243, B) float32 table of every 5-trit pattern scored against each 5-wide query slice; each row's score is the sum of 78 table rows picked by its 5-trit group codes (pair code // 243 and % 243, split once into a separate uint8 array).
- `lut5s`: same table, but the 5-trit split happens inside each chunk from the pair codes, so no second index.
- `lut10`: per batch, a (39, 59,049, B) table, one pick per pair code, 39 per row.

Table build time is inside each timed sweep. 5 timed repeats per mode; every sweep's top-10 compared to the int8 GEMM reference at the same batch. numpy 2.5.4, 8-vCPU shared box, MemAvailable about 6.5 GB at start, python3 and numpy only, no network, nothing bound. A 65,536-row smoke run came first (`python3 bench.py 65536 7 1 ...`, exact top-10 in all six modes). Command, run in a scratch folder (script not committed, about 80 lines, imports `tern`, `enc_pair`, `V5i`, `V10` from the 2026-10-30 14:55 scratch script):

```
python3 bench.py 1048576 202611010255 5 int8:64,int8:128,unpack:64,unpack:128,lut5:64,lut5:128,lut5s:64,lut10:64   # N, seed, repeats, mode:batch
```

## Results

Median of 5 sweeps over 1,048,576 rows in ms, spread (max minus min) in brackets. "Per 64 queries" divides the B = 128 sweeps by 2 so batches compare per query. Top-10 matched the int8 reference exactly (100.00%) in every mode.

| mode | B | sweep ms | per 64 queries ms | table build ms | table bytes | vs unpack B=64 |
|---|---|---|---|---|---|---|
| int8 GEMM | 64 | 103 (10) | 103 | 0 | 0 | 1.19x |
| int8 GEMM | 128 | 166 (7) | 83 | 0 | 0 | 1.48x |
| pair unpack + GEMM (current) | 64 | 123 (30) | 123 | 0 | 0 | 1.00x |
| pair unpack + GEMM | 128 | 208 (19) | 104 | 0 | 0 | 1.18x |
| pair lut5, split index | 64 | 809 (24) | 809 | 6 | 4,852,224 | 0.15x |
| pair lut5, split index | 128 | 1,507 (46) | 754 | 16 | 9,704,448 | 0.16x |
| pair lut5, split in chunk | 64 | 830 (42) | 830 | 7 | 4,852,224 | 0.15x |
| pair lut10 | 64 | 1,734 (757) | 1,734 | 1,123 | 589,545,216 | 0.07x |

The current pair path costs 123 ms per million rows at B = 64, which scales to about 246 ms for 2,000,000 rows and matches the 18:55 floor of about 260 ms, so the in-memory model is sound.

Scoring from the codes with lookup tables is 6.6x to 14x slower than unpacking and multiplying. The 5-trit table is small (4.9 MB at B = 64) but needs 78 row picks of B floats per index row, and numpy pays per pick in memory traffic, while BLAS does the 384-wide dot products in registers. The 10-trit table halves the picks but is 590 MB, takes 1.1 s to build per batch, and misses cache on every pick. Splitting halves inside the chunk costs only 21 ms over a stored half index, so the second index is not worth its 82 MB either way.

Batching 128 queries per chunk helps every GEMM path a little: pair unpack goes from 123 to 104 ms per 64 queries (1.18x) and int8 from 103 to 83 ms (1.24x), because each chunk's unpack and int8-to-float cast is shared by twice the queries. That is short of the 1.5x bar. In memory, int8 is still 1.19x to 1.25x faster than pair codes per query, so pair codes win in `night` only through the 4.9x smaller reads (the 18:55 result), not through compute.

## Verdict

**Not viable.** Neither lever cuts the pair-code compute floor by 1.5x. Lookup-table scoring is 6.6x slower (lut5, 809 vs 123 ms per million rows) or 14x slower (lut10) at exact top-10, so it is not worth it in numpy. Batching 128 queries per chunk is a real but small gain, 1.18x for pair codes (104 vs 123 ms per 64 queries), and is worth taking only where `night` already has 128 queries queued. The pair advantage stays a storage advantage: keep unpack + GEMM, keep 8 readers + 8 scorers.

Caveats: in memory only, so these numbers are the compute floor, not end-to-end; 1,048,576 rows rather than 2,000,000 (scaled where compared); synthetic embeddings, one seed; shared 8-vCPU box, with a wide lut10 spread (757 ms) from its table build; numpy `take` stands in for a native gather kernel, which could change the lut5 picture (a SIMD pshufb-style lookup is how native ternary kernels win), so this verdict holds for the Python path only.

## Pair files this wave

- No pair file for the 02:05 wave reached this run, and no new roadmap item is named; the roadmap blobs have had nothing uncovered since 2026-10-24, and the ledger will confirm the Chief's call. In JuniorLLM the only commit in the last 6 hours is [4772a84](https://github.com/cloudcover95/JuniorLLM/commit/4772a84) (02:52 MT, cloudcover95, one file `rails/dst/dst_window_check.py`, no roadmap item named), which this note does not touch; this note adds one new file under `junior_osai/vault/findings/` only. The ask/night side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

With compute fixed at unpack + GEMM, does decoding pair codes straight to float32 with a 59,049 x 10 float32 table (skipping the int8 take and the separate cast), or decoding to float16 and scoring in float16, cut the 123 ms per million rows unpack cost by enough to bring pair codes level with int8 in memory?

close: 16357d4efc2d5ac8 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-31_1855)
