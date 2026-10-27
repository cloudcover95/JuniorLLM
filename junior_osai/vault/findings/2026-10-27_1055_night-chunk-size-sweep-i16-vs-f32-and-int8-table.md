---
local: true
date: 2026-10-27
wave: "2026-10-27 10:05 (bench 10:55, ran 11:00)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl ask` / `night` / `security` (64-ask `night` sweep over a resident base-3 ternary index at 100k and 500k docs)"
verdict: not-viable
tags: [osai, vault, findings, ternary, base3, retrieval, juniorctl, night, lut, int16, int8, topk, chunked, chunk-size, bench]
---

# juniorctl night: chunk-size sweep for the int16 table, and an int8 table with int16 accumulation

## Question

The 05:55 note ([[2026-10-27_0555_night-int16-chunked-topk-vs-f32-base3-sweep]]) found the int16 shared table 1.45x to 1.67x faster than f32 at 8,192-row chunks. Its next question: does that edge hold when the chunk size moves (2k, 4k, 16k, 32k rows), and does an int8 table accumulated in int16 (half the table bytes again) stay above 0.99 recall@10 while clearing 1.5x over the int16 table? No Quant Chief assignment or pair report for the 10:05 wave reached this run, so it stays on the ask/night line.

## Method

Same synthetic clustered unit embeddings as the 10/26 10:55 to 10/27 05:55 runs (384 dims, 256 topic centers plus 0.9 noise, absmean-threshold ternarization at 0.5 mean|x|), packed 5 trits per byte in base 3 (77 B/row), at 100,000 and 500,000 docs. 128 normalized queries as two 64-ask batches, three passes, so six timed batches per cell; table build and top-10 selection are inside the timed path. Every variant uses the fused per-chunk top-10 and a final merge.

Three tables per chunk size: f32 (77 x 243 x 64), int16 with the 05:55 overflow-proof per-ask scale, and int8 scaled so each ask's largest entry is 127, accumulated in int16 (the worst row sum is 77 x 127 = 9,779, so it cannot overflow). Recall@10 is against the exact top 10 from exact ternary dots; top-1 counts asks whose best returned doc has the exact best score. numpy 2.5.4, one thread, same Intel Xeon box, two seeds, python3 and numpy only, no network.

Command, run in a scratch folder (not committed):

```
OPENBLAS_NUM_THREADS=1 python3 bench.py          # seed 202610271055
OPENBLAS_NUM_THREADS=1 python3 bench.py 777
```

New kernels (`table`, `table16` and the index build are unchanged from the 05:55 note):

```python
def table8(Q):                        # int8 entries, per-ask scale to 127
    T=table(Q); s=127.0/np.abs(T).max((0,1)); return np.round(T*s).astype(np.int8), s
def scan(I,Tab,adt):                  # adt = float32 or int16 accumulator
    acc=Tab[0][I[0]].astype(adt)
    for j in range(1,I.shape[0]): acc+=Tab[j][I[j]]
    return acc
# topk_chunk(I,Tab,adt,CH): scan each CH-row slice, argpartition top-10 per chunk, merge at the end
```

## Results

Median ms per ask, seed 202610271055 / seed 777 (p10 to p90 spread stayed between 0.04 and 2.1 ms; full lines in the run output).

| N | chunk | f32 | int16 table | int8 table, int16 acc | int16 vs f32 | int8 vs int16 |
|---|---|---|---|---|---|---|
| 100k | 2k | 2.94 / 2.87 | **2.20 / 2.25** | 2.49 / 2.33 | 1.33x / 1.28x | 0.89x / 0.97x |
| 100k | 4k | 3.38 / 3.42 | 2.35 / 2.45 | 2.49 / 2.48 | 1.44x / 1.39x | 0.94x / 0.99x |
| 100k | 8k | 4.65 / 4.68 | 2.84 / 2.94 | 2.84 / 2.87 | 1.64x / 1.59x | 1.00x / 1.02x |
| 100k | 16k | 5.08 / 5.07 | 3.51 / 3.57 | 3.14 / 3.19 | 1.45x / 1.42x | 1.12x / 1.12x |
| 100k | 32k | 5.17 / 5.16 | 3.80 / 3.78 | 3.63 / 3.32 | 1.36x / 1.36x | 1.05x / 1.14x |
| 500k | 2k | 15.3 / 14.0 | **11.3 / 11.2** | 12.5 / 12.0 | 1.35x / 1.25x | 0.91x / 0.93x |
| 500k | 4k | 17.3 / 16.4 | 11.7 / 11.5 | 12.1 / 11.8 | 1.47x / 1.42x | 0.97x / 0.98x |
| 500k | 8k | 25.3 / 24.6 | 14.4 / 15.2 | 14.6 / 15.1 | 1.76x / 1.62x | 0.98x / 1.01x |
| 500k | 16k | 26.4 / 25.9 | 17.8 / 17.9 | 16.2 / 16.2 | 1.48x / 1.44x | 1.10x / 1.11x |
| 500k | 32k | 26.0 / 25.5 | 18.3 / 18.5 | 16.6 / 16.9 | 1.42x / 1.38x | 1.10x / 1.10x |

Best chunk for each table against best chunk for f32 (all three are fastest at 2k):

| N | f32 best | int16 best | int8 best | int16 vs f32 | int8 vs int16 |
|---|---|---|---|---|---|
| 100k | 2.94 / 2.87 (2k) | 2.20 / 2.25 (2k) | 2.49 / 2.33 (2k) | 1.33x / 1.28x | 0.88x / 0.97x |
| 500k | 15.3 / 14.0 (2k) | 11.3 / 11.2 (2k) | 12.1 / 11.8 (4k) | 1.35x / 1.25x | 0.93x / 0.95x |

| item | value |
|---|---|
| table bytes, 64 asks: f32 / int16 / int8 | 4,790,016 / 2,395,008 / 1,197,504 |
| base-3 index at 100k / 500k | 7,700,000 / 38,500,000 |
| recall@10: f32 / int16 / int8 | 1.000 / 0.996 to 0.998 / 0.973 to 0.981 |
| top-1, all variants | 128/128 |
| max score error, int16 / int8 (score units, dot range about +/-20) | 6.2e-3 to 6.9e-3 / 4.5e-2 to 4.7e-2 |

## Verdict

**Not viable**, and it corrects the 05:55 number. The int16 table only beat f32 by 1.5x at 8k chunks (1.59x to 1.76x here, matching 05:55) because 8k happens to be a bad chunk size for f32: f32 at 8k is 1.6x to 1.8x slower than f32 at 2k. With each side at its own best chunk (2k for both), int16 is 1.25x to 1.35x faster than f32, under the 1.5x bar. The table-byte halving is real (4.8 MB to 2.4 MB) but the per-ask accumulator at 2k rows is only 0.5 MB in f32, so it no longer buys much.

The int8 table halves table bytes again but is no faster at the chunk sizes that matter (0.88x to 0.97x of int16 at its best chunk; it only wins, by about 1.1x, at 16k and 32k where everything is slower), and recall@10 drops to 0.973 to 0.981, below the 0.99 bar. Not worth it.

What does carry over: 2k-row chunks are the right size for every table here, and switching the `juniorctl night` rule from 8k to 2k chunks is about 1.3x faster per ask on its own (int16 at 8k 2.84 to 15.2 ms against int16 at 2k 2.20 to 11.3 ms). Updated rule: base-3 5-trit index, 64-ask int16 table with the overflow-proof per-ask scale, 2,048-row chunks with per-chunk top-10. Its edge over the same sweep in f32 is about 1.3x, so the 05:55 "viable" should be read as viable only against an f32 sweep left at 8k chunks.

Caveats: synthetic embeddings only, one thread, one machine.

## Pair files this wave

- No pair report or Quant Chief assignment for the 10:05 wave reached this run, and the fresh push clone shows no JuniorLLM push since 06:13 MT (tip [1f56b09](https://github.com/cloudcover95/JuniorLLM/commit/1f56b09), the 05:55 bench note itself), so there is no 10:05 pair file to link and no path overlap with Trit Pack, Handshake or an LLC bot. The ask side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Below 2k, is there a floor (512 and 1,024-row chunks), and does the base-3 int16 sweep at its best chunk clear 1.5x over a plain int8 dense matmul baseline (unpacked int8 trits times int8-quantized queries in one `@` per chunk), which is the honest non-LUT competitor for `night`?

close: 6bacd50cfa537792 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-27_0555)
