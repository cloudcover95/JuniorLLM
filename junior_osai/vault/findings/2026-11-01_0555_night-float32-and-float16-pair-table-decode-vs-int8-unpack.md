---
local: true
date: 2026-11-01
wave: "2026-11-01 05:05 (bench 05:55, ran 06:00 to 06:03)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`night` sweeping the packed ask index)"
verdict: not-viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, night, ask-index, unpack, float32-table, float16, compute-floor, bench]
---

# juniorctl night unpack cost: decoding pair codes straight to float32 or float16 vs the current int8 table + cast

## Question

The 02:55 note ([[2026-11-01_0255_night-lut-scoring-and-128-query-batch-pair-codes-compute-floor]]) left unpack + GEMM as the pair-code compute path at about 123 ms per million rows (B = 64), against about 103 ms for int8 rows. Does decoding the uint16 pair codes straight to float32 through a 59,049 x 10 float32 table (no int8 take followed by a separate cast), or through a float16 table, cut that unpack cost by 1.5x or more, or at least bring pair codes level with int8 in memory?

## Method

In-memory sweep over a synthetic 1,048,576-row x 384-dim index, built with the 2026-10-30 14:55 encoders unchanged (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row). 64 queries, 8 scoring threads over 8,192-row chunks (128 chunks), OpenBLAS pinned to 1 thread, top-10 per query merged across chunks, every sweep checked against the int8 GEMM reference. Modes:

- `int8`: int8 rows to float32, `Q @ X.T` (reference).
- `unpack`: current pair path, `np.take` on the int8 pair table (590 KB), slice to 384, `astype(float32)`, `Q @ X.T`.
- `f32slice`: `np.take` on a float32 pair table (2.4 MB), slice to 384, `Q @ X.T` (no cast).
- `f32pad`: same float32 take, no slice; queries zero-padded to 390 so the gathered block goes to BLAS as-is.
- `i8pad`: int8 take + cast, no slice, padded queries.
- `f16cast`: `np.take` on a float16 table (1.2 MB), cast to float32, padded GEMM.
- `f16gemm`: float16 take and float16 `Qh @ X.T` (numpy has no BLAS for float16; smoke size only).

7 timed repeats per mode, two seeds (202611010555, then 202611010556 with modes reordered and repeated to expose drift). numpy 2.5.4, 8-vCPU shared box, MemAvailable about 2.4 GB at start, python3 and numpy only, no network, nothing bound. A 65,536-row smoke run came first (`python3 bench.py 65536 7 3 int8,unpack,f32slice,f32pad,f16cast,f16gemm,i8pad`, top-10 exact in every mode except f16gemm at 99.22%). Script not committed (about 60 lines, imports `tern`, `enc_pair`, `V10` from the 2026-10-30 14:55 scratch script). Commands, run in a scratch folder:

```
python3 bench.py 1048576 202611010555 7 int8,unpack,f32slice,f32pad,f16cast,i8pad,int8,unpack     # N, seed, repeats, modes
python3 bench.py 1048576 202611010556 7 unpack,f32slice,int8,f32pad,unpack,f32slice
```

## Results

Median of 7 sweeps over 1,048,576 rows in ms, spread (max minus min) in brackets. Where a mode ran twice in one process both medians are shown. Top-10 matched the int8 reference exactly (100.00%) in every full-size mode.

| mode | gather table bytes | seed ...555 ms | seed ...556 ms | vs unpack (same seed) | vs int8 (same seed) |
|---|---|---|---|---|---|
| int8 rows + cast | none (384 B/row read) | 92 (18), 84 (8) | 85 (6) | 1.25x faster | 1.00x |
| pair unpack, int8 table + cast (current) | 590,490 | 112 (16), 109 (15) | 102 (13), 104 (7) | 1.00x | 0.80x |
| pair, float32 table, slice | 2,361,960 | 97 (5) | 98 (5), 102 (5) | 1.13x / 1.03x | 0.91x / 0.85x |
| pair, float32 table, padded GEMM | 2,361,960 | 98 (5) | 113 (51) | 1.12x / 0.91x | 0.90x / 0.75x |
| pair, int8 table + cast, padded GEMM | 590,490 | 99 (9) | not run | 1.12x | 0.89x |
| pair, float16 table + cast | 1,180,980 | 319 (30) | not run | 0.35x | 0.28x |
| pair, float16 GEMM (smoke, 65,536 rows, scaled) | 1,180,980 | 21,095 per 1M | not run | 0.01x | 0.01x |

The float32 table saves the int8-to-float cast, but the gather itself now moves 4x the bytes per trit (40 B per pair code instead of 10), so the net gain is small: 1.13x on the first seed (97 vs 110 ms, average of the two unpack runs) and 1.03x on the second (100 vs 103 ms). Its spread is tighter (5 ms vs 13 to 16 ms), which is the clearest effect. Skipping the 384 slice by padding the queries to 390 helps the int8 table about as much (99 ms) and does nothing extra for float32 once the slice copy is gone; the 51 ms spread on the second seed's padded run is box noise, not a trend. float16 is a loss: numpy's float16 to float32 cast is slow (319 ms, 2.9x worse), and float16 GEMM has no BLAS path at all (about 21 s per million rows).

So the pair path stays 10% to 15% behind int8 in memory however it is decoded in numpy, which matches the 02:55 finding that the pair advantage in `night` is the 4.9x smaller read, not compute.

## Verdict

**Not viable.** The float32 pair table is the best decode tried, but it improves the current unpack by only 1.03x to 1.13x (97 to 100 ms vs 103 to 110 ms per million rows at exact top-10), well short of the 1.5x bar, and it still trails int8 rows (84 to 92 ms). float16 decode is 2.9x worse and float16 GEMM about 190x worse. It is a cheap, safe tweak worth taking if `night` is touched anyway (2.4 MB table, tighter timing), but not a ternary efficiency win.

Caveats: in memory only, so this is the compute floor, not end-to-end; synthetic embeddings; two seeds; shared 8-vCPU box with about 2.4 GB free, so ms-level differences under 10% sit inside the noise; numpy `take` stands in for a native gather, and a native kernel that expands trits in registers would change the picture.

## Pair files this wave

- No pair file for the 05:05 wave reached this run, and no new roadmap item is named; the roadmap blobs have had nothing uncovered since 2026-10-24, and the ledger will confirm the Chief's call. JuniorLLM's tip at clone time was this lane's own 02:55 note, [3be741f](https://github.com/cloudcover95/JuniorLLM/commit/3be741f); the only other commit in the last 6 hours is [4772a84](https://github.com/cloudcover95/JuniorLLM/commit/4772a84) (02:52 MT, `rails/dst/dst_window_check.py`), which this note does not touch; this note adds one new file under `junior_osai/vault/findings/` only. The ask/night side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

With the numpy compute floor settled (unpack + GEMM, about 100 ms per million rows, int8 about 10% to 15% faster in memory), how does end-to-end `night` scale when the index outgrows page cache: at 8,000,000 rows on a modeled 1.6 GB/s device with 8 readers + 8 scorers, does the pair-code lead over int8 hold at about 2x, or does the 4.9x read saving grow once int8 can no longer stay cached?

close: 91cd359ef145e24c (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-01_0255)
