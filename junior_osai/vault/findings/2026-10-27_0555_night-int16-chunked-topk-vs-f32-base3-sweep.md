---
local: true
date: 2026-10-27
wave: "2026-10-27 05:05 (bench 05:55, ran 06:10)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl ask` / `night` / `security` (64-ask `night` sweep over a resident base-3 ternary index at 100k to 500k docs)"
verdict: viable
tags: [osai, vault, findings, ternary, base3, retrieval, juniorctl, night, batch, lut, int16, topk, chunked, bench]
---

# juniorctl night: int16 accumulation and chunked top-k for the 64-ask base-3 sweep

## Question

The 02:55 note ([[2026-10-27_0255_night-batch-size-sweep-8bit-vs-base3-5trit-shared-lut]]) fixed the `night` sweep at a 64-ask shared table over the base-3 5-trit index and saw it slow down once the N x 64 f32 accumulator fell out of L2. Its next question: does a top-k step fused into the scan (one argpartition per 8,192-row chunk instead of holding the full accumulator) keep the sweep flat from 100k to 500k docs, and does int16 accumulation of a scaled table clear 1.5x over the f32 accumulator at acceptable top-1 agreement? No Quant Chief assignment or pair report for the 05:05 wave reached this run, so it stays on the ask/night line.

## Method

Same synthetic clustered unit embeddings as the 10:55 to 02:55 runs (384 dims, 256 topic centers plus 0.9 noise, absmean-threshold ternarization at 0.5 mean|x|), packed 5 trits per byte in base 3 (77 B/row). 100,000, 250,000 and 500,000 docs. 128 normalized queries drawn from the same topic mix, run as two 64-ask batches, three passes, so six timed batches per variant; per-ask time is batch time divided by 64, and the table build and the top-10 selection are inside the timed path.

The int16 table scales each ask's 77 x 243 table by 32000 / (sum over groups of the largest |entry|), so no row sum can overflow int16; scores are divided back by that scale only for the error check. Recall@10 is against the exact top 10 from exact ternary dots (int8 trits times f32 query); top-1 counts asks whose best returned doc has the exact best score. numpy 2.5.4, one thread, Intel Xeon (16 MiB L2 across 8 instances, 320 MiB L3), two seeds, python3 and numpy only, no network.

Command, run in a scratch folder (not committed):

```
OPENBLAS_NUM_THREADS=1 python3 bench.py          # seed 202610270555
OPENBLAS_NUM_THREADS=1 python3 bench.py 777
```

Core kernels (`scan` is the same gather-and-add loop as the 02:55 note):

```python
def table16(Q):                       # 77 x 243 x 64, int16, per-ask scale with no overflow possible
    T=table(Q); s=32000.0/np.abs(T).max(1).sum(0)
    return np.round(T*s).astype(np.int16), s
def topk_chunk(I,Tab):                # I: 77 x N uint8; one 8,192-row accumulator at a time
    N=I.shape[1]; ci=[]; cv=[]
    for s in range(0,N,CH):
        acc=scan(I[:,s:s+CH],Tab); p=np.argpartition(-acc,K,axis=0)[:K]
        ci.append(p+s); cv.append(np.take_along_axis(acc,p,0))
    ci=np.concatenate(ci); cv=np.concatenate(cv); p=np.argpartition(-cv.astype(np.float32),K,axis=0)[:K]
    return np.take_along_axis(ci,p,0)
```

`topk_full` is `scan` over all N rows followed by one argpartition.

## Results

Median ms per ask (p10 to p90 spread in brackets). The last column is the head-to-head this note asks about: int16 against f32, both chunked.

| N | variant | seed 202610270555 | seed 777 | recall@10 | top-1 |
|---|---|---|---|---|---|
| 100k | f32 full | 5.88 (0.43) | 5.61 (0.35) | 1.000 | 128/128 |
| 100k | f32 chunk 8k | 5.25 (0.23) | 4.91 (0.08) | 1.000 | 128/128 |
| 100k | i16 full | 4.73 (0.35) | 4.55 (0.34) | 0.995 to 0.998 | 128/128 |
| 100k | i16 chunk 8k | **3.32 (0.13)** | **3.22 (0.10)** | 0.996 to 0.998 | 128/128 |
| 250k | f32 full | 16.9 (1.9) | 16.3 (1.9) | 1.000 | 128/128 |
| 250k | f32 chunk 8k | 12.2 (0.57) | 12.0 (0.22) | 1.000 | 128/128 |
| 250k | i16 full | 11.5 (1.4) | 12.0 (1.7) | 0.995 to 0.997 | 128/128 |
| 250k | i16 chunk 8k | **7.76 (0.21)** | **8.28 (0.72)** | 0.995 to 0.997 | 128/128 |
| 500k | f32 full | 59.1 (9.8) | 60.7 (10.0) | 1.000 | 128/128 |
| 500k | f32 chunk 8k | 25.3 (0.96) | 26.5 (1.8) | 1.000 | 128/128 |
| 500k | i16 full | 32.2 (7.3) | 35.2 (3.1) | 0.995 | 128/128 |
| 500k | i16 chunk 8k | **16.2 (1.05)** | **15.9 (0.41)** | 0.994 to 0.995 | 128/128 |

| N | i16 chunk vs f32 chunk | i16 chunk vs f32 full | f32 chunk per 100k rows | f32 full per 100k rows |
|---|---|---|---|---|
| 100k | 1.58x / 1.52x | 1.77x / 1.74x | 5.25 / 4.91 ms | 5.88 / 5.61 ms |
| 250k | 1.57x / 1.45x | 2.18x / 1.97x | 4.87 / 4.80 ms | 6.76 / 6.52 ms |
| 500k | 1.57x / 1.67x | 3.65x / 3.82x | 5.07 / 5.30 ms | 11.8 / 12.1 ms |

| item | bytes |
|---|---|
| base-3 index at 500k | 38,500,000 (77 B/row) |
| f32 full accumulator at 500k, 64 asks | 128,000,000 (122 MiB) |
| int16 full accumulator at 500k | 64,000,000 (61 MiB) |
| chunk accumulator, f32 / int16 | 2,097,152 / 1,048,576 |
| int16 max score error (score units, dot range about +/-20) | 6.2e-3 to 7.2e-3 |

## Verdict

**Viable.** The int16 table with chunked accumulation runs 1.45x to 1.67x faster per ask than the same chunked scan in f32 (median 1.57x; five of six cells at 1.52x or better), halves the accumulator bytes, and keeps top-1 at 128/128 on both seeds with recall@10 of 0.994 to 0.998. This one is a ternary result: the int16 path is safe only because every row score is a sum of 77 bounded trit-group partials, so a per-ask scale can rule out overflow. Against today's f32 full-accumulator sweep it is 1.74x faster at 100k and 3.65x to 3.82x faster at 500k.

The fused chunked top-k answers the first half of the question on its own: in f32 it holds the sweep flat at 4.8 to 5.3 ms per 100k rows from 100k to 500k, while the full accumulator degrades to about 12 ms per 100k rows at 500k (2.29x to 2.33x slower), because the 122 MiB accumulator leaves cache. Chunking isn't a ternary gain by itself, so it doesn't count toward the verdict.

Caveats: synthetic embeddings only, one thread, and the 0.2 to 0.6% recall@10 loss is ties near the 10th place, not missed top hits. Updated rule for `juniorctl night`: base-3 5-trit index, 64-ask shared table quantized to int16 with a per-ask overflow-proof scale, 8,192-row chunks with per-chunk top-10, merged at the end.

## Pair files this wave

- No pair report or Quant Chief assignment for the 05:05 wave reached this run, and the fresh clone shows no new JuniorLLM push since 03:03 MT (tip [ca1721e](https://github.com/cloudcover95/JuniorLLM/commit/ca1721e), the 02:55 bench note itself), so there is no 05:05 pair file to link and no path overlap with Trit Pack, Handshake or an LLC bot. The ask side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Does the int16 chunked sweep hold its 1.5x edge over f32 when the chunk size moves (2k, 4k, 16k, 32k rows), and does an int8 table with int16 accumulation (half the table bytes again, 77 x 243 x 64 x 1 B = 1.2 MB) stay above 0.99 recall@10 while clearing 1.5x over the int16 table?

close: 447180523b1cf9ed (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-27_0255)
