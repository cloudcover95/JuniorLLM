---
local: true
date: 2026-10-27
wave: "2026-10-27 14:05 (bench 14:55, ran 14:58)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl ask` / `night` / `security` (64-ask `night` sweep over a resident base-3 ternary index at 100k and 500k docs)"
verdict: not-viable
tags: [osai, vault, findings, ternary, base3, retrieval, juniorctl, night, lut, int16, dense-baseline, sgemm, chunk-size, bench]
---

# juniorctl night: chunk floor below 2k, and the base-3 LUT sweep against a dense int8-stored matmul baseline

## Question

The 10:55 note ([[2026-10-27_1055_night-chunk-size-sweep-i16-vs-f32-and-int8-table]]) settled on 2,048-row chunks and an int16 shared table. Its next question: is there a floor below 2k (512 and 1,024-row chunks), and does the base-3 int16 LUT sweep at its best chunk clear 1.5x over the honest non-LUT competitor, a plain dense matmul over unpacked int8 trits? This run adds a third path, unpacking the base-3 bytes to f32 per chunk and calling one `@`, to see whether ternary storage can keep the dense speed. No Quant Chief assignment or pair report for the 14:05 wave reached this run, so it stays on the ask/night line.

## Method

Same synthetic clustered unit embeddings as the 10/26 10:55 to 10/27 10:55 runs (384 dims, 256 topic centers plus 0.9 noise, absmean-threshold ternarization at 0.5 mean|x|), packed 5 trits per byte in base 3 (77 B/row), at 100,000 and 500,000 docs. 128 normalized queries as two 64-ask batches, three passes, so six timed batches per cell (two for the slow numpy integer matmul); table build, any per-chunk unpack or cast, and top-10 selection are all inside the timed path. Every variant uses per-chunk top-10 and a final merge. Recall@10 is against the exact top 10 from exact ternary dots; top-1 counts asks whose best returned doc has the exact best score. numpy 2.5.4, one thread, same Intel Xeon box, two seeds, python3 and numpy only, no network.

Variants:

- `lut f32` / `lut i16`: the 10:55 shared-table scan over the base-3 index (int16 with the overflow-proof per-ask scale), chunks 512, 1,024 and 2,048.
- `dense f32gemm`: trits stored unpacked as int8 (384 B/row), each chunk cast to f32 and multiplied by the f32 query block in one BLAS `@`.
- `base3 unpack+gemm`: base-3 index (77 B/row), each chunk decoded through the 243 x 5 trit table to f32, then the same `@`.
- `dense i8/i32`: int8 trits times int8-quantized queries with int32 accumulation (numpy integer matmul, no BLAS).

Command, run in a scratch folder (not committed):

```
OPENBLAS_NUM_THREADS=1 python3 bench.py          # seed 202610271455
OPENBLAS_NUM_THREADS=1 python3 bench.py 777
```

New kernels (`table`, `table16`, `scan` and the index build are unchanged from the 10:55 note):

```python
def dense_topk(T8,Q,CH):                 # T8: N x 384 int8 trits
    Qf=np.ascontiguousarray(Q.T)
    for s in range(0,N,CH): acc=T8[s:s+CH].astype(np.float32)@Qf; ...   # per-chunk top-10, merge
def unpack_topk(It,Q,CH):                # It: N x 77 base-3 bytes; V5: 243 x 5 f32 trit table
    Qf=np.ascontiguousarray(np.pad(Q,((0,0),(0,1))).T)
    for s in range(0,N,CH): acc=V5[It[s:s+CH]].reshape(-1,385)@Qf; ...
```

## Results

Median ms per ask, seed 202610271455 / seed 777 (p10 to p90 spread 0.02 to 2.1 ms; full lines in the run output).

| N | variant | chunk 512 | chunk 1k | chunk 2k | chunk 4k | chunk 8k | recall@10 |
|---|---|---|---|---|---|---|---|
| 100k | lut f32 | 3.29 / 3.87 | 2.88 / 3.07 | 2.68 / 2.88 | | | 1.000 |
| 100k | lut i16 | 2.68 / 2.98 | 2.33 / 2.54 | **2.21 / 2.29** | | | 0.995 to 0.997 |
| 100k | base3 unpack+gemm | | 2.10 / 2.18 | 2.10 / 2.22 | 2.35 / 2.31 | | 1.000 |
| 100k | dense f32gemm (int8 stored) | | 1.19 / 1.19 | **1.13 / 1.14** | | 1.77 / 1.74 | 1.000 |
| 100k | dense i8/i32 numpy | | | 15.2 / 16.3 | | | 0.989 to 0.990 |
| 500k | lut f32 | 17.2 / 16.7 | 14.7 / 15.1 | 14.4 / 13.8 | | | 1.000 |
| 500k | lut i16 | 13.3 / 13.7 | 11.4 / 11.6 | **10.5 / 10.6** | | | 0.993 to 0.998 |
| 500k | base3 unpack+gemm | | 10.5 / 10.7 | 10.4 / 10.6 | 11.4 / 11.6 | | 1.000 |
| 500k | dense f32gemm (int8 stored) | | 5.66 / 5.73 | **5.53 / 5.60** | | 8.42 / 8.57 | 1.000 |
| 500k | dense i8/i32 numpy | | | 77.9 / 77.3 | | | 0.987 to 0.989 |

Best chunk against best chunk (top-1 was 128/128 for every variant):

| N | dense f32gemm vs lut i16 | dense f32gemm vs base3 unpack+gemm | base3 unpack+gemm vs lut i16 | lut i16 vs dense i8/i32 |
|---|---|---|---|---|
| 100k | 1.95x / 2.01x faster | 1.85x / 1.92x faster | 1.05x / 1.05x | 6.9x / 7.1x faster |
| 500k | 1.89x / 1.89x faster | 1.88x / 1.89x faster | 1.01x / 1.00x | 7.5x / 7.3x faster |

| storage | bytes per row | 100k | 500k |
|---|---|---|---|
| base-3, 5 trits per byte | 77 | 7,700,000 | 38,500,000 |
| int8 trits, unpacked | 384 | 38,400,000 | 192,000,000 |
| f32 trits (not built) | 1,536 | 153,600,000 | 768,000,000 |

## Verdict

**Not viable** on speed, and this reverses the direction of the last three notes. Against the honest dense competitor (trits kept as int8, cast per 2k chunk, one BLAS `@`), the base-3 int16 LUT sweep is 1.89x to 2.01x *slower* at both sizes and both seeds, with slightly lower recall (0.993 to 0.998 against 1.000). The LUT only looked good because every earlier note compared it to other LUT tables. Decoding the base-3 bytes back to f32 per chunk and using the same `@` matches the LUT (1.00x to 1.05x) with exact recall, but still runs about 1.9x behind int8 storage, so the decode costs as much as the LUT scan. Plain numpy int8 times int8 with int32 accumulation is the slowest path by far (7x behind the LUT, 14x behind the cast-and-sgemm path) because numpy integer matmul has no BLAS.

Chunk floor: 2,048 rows stays best. 512 and 1,024-row chunks are 5% to 27% slower for every LUT variant, and the dense paths are flat between 1k and 2k.

What ternary still buys here is memory, not speed: the base-3 index is 5.0x smaller than int8 trits (38.5 MB against 192 MB at 500k) at equal recall through the unpack path. That is a trade, about 1.9x slower per ask for 5x less resident memory, and it was already known from the packing notes, so it is not a new gain. Updated `night` rule: if the 500k index fits in RAM as int8 (192 MB), store int8 and use per-2k-chunk f32 `@`; keep base-3 plus `unpack+gemm` (exact, no table) only when resident memory is the binding limit, and drop the int16 LUT path, since `unpack+gemm` is as fast with exact recall.

Caveats: synthetic embeddings only, one thread, one machine; with more threads the BLAS path would likely pull further ahead.

## Pair files this wave

- No pair report or Quant Chief assignment for the 14:05 wave reached this run, and the fresh push clone shows no JuniorLLM push since 11:06 MT (tip [8d83985](https://github.com/cloudcover95/JuniorLLM/commit/8d83985), the 10:55 bench note itself), so there is no 14:05 pair file to link and no path overlap with Trit Pack, Handshake or an LLC bot. The ask side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Can a fused decode cut the unpack cost enough to close the 1.9x gap to int8 storage: decode two base-3 bytes at once through a 59,049-row x 10 int8 table (590 KB) into an int8 chunk, then cast and `@`, and does that beat 1.5x over today's `unpack+gemm` at 2k chunks while keeping the 5x memory saving?

close: b6cb23af67de712b (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-27_1055)
