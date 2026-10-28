---
local: true
date: 2026-10-27
wave: "2026-10-27 18:05 (bench 18:55, ran 18:59)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl ask` / `night` / `security` (64-ask `night` sweep over a resident base-3 ternary index at 100k and 500k docs)"
verdict: not-viable
tags: [osai, vault, findings, ternary, base3, retrieval, juniorctl, night, fused-decode, pair-table, sgemm, bench]
---

# juniorctl night: fused two-byte base-3 decode against single-byte unpack and int8 storage

## Question

The 14:55 note ([[2026-10-27_1455_night-base3-lut-vs-dense-int8-gemm-baseline]]) found that decoding the base-3 index to f32 per 2k chunk and calling one BLAS `@` is exact but about 1.9x slower than keeping trits as int8. Its next question: does a fused decode, two base-3 bytes at once through a 59,049-row x 10 trit table, cut the unpack cost by at least 1.5x over today's `unpack+gemm` while keeping the 5x memory saving? No new Quant Chief assignment for the 18:05 wave reached this run (Chief tips file at 18:11 MT still shows JuniorLLM tip 447e548), so it stays on the ask/night line.

## Method

Same synthetic clustered unit embeddings as the 10/26 to 10/27 night runs (384 dims, 256 topic centers plus 0.9 noise, absmean-threshold ternarization at 0.5 mean|x|), base-3 packed 5 trits per byte (77 B/row), at 100,000 and 500,000 docs. For the pair paths the 77 bytes are pre-combined into 39 uint16 pair codes `b0*243 + b1` (last byte padded; 78 B/row), and an assert checks the pair decode reproduces the int8 trits exactly. 128 normalized queries as two 64-ask batches, four passes, so eight timed batches per cell; decode, cast, `@` and per-chunk top-10 plus merge are all inside the timed path. Chunk is fixed at 2,048 rows (the 14:55 floor). numpy 2.5.4, one thread, same Intel Xeon box, two seeds, python3 and numpy only, no network.

Variants:

- `dense f32gemm`: int8-stored trits (384 B/row), each chunk cast to f32, one `@` (the 14:55 winner).
- `base3 unpack f32`: 14:55 baseline, gather through the 243 x 5 f32 table.
- `base3 unpack int8+cast`: same gather through an int8 243 x 5 table, then cast.
- `pair decode int8+cast`: gather through the 59,049 x 10 int8 table (590 KB), then cast to f32.
- `pair decode f32`: gather through a 59,049 x 10 f32 table (2.36 MB), no cast.

Command, run in a scratch folder (not committed):

```
OPENBLAS_NUM_THREADS=1 python3 bench.py          # seed 202610271855
OPENBLAS_NUM_THREADS=1 python3 bench.py 777
```

Kernel core:

```python
V10i=np.concatenate([np.repeat(V5i,243,0),np.tile(V5i,(243,1))],1)   # 59049 x 10 int8
Pp=Ip[:,0::2]*243+Ip[:,1::2]                                         # N x 39 uint16 pair codes
for s in range(0,N,2048):
    acc=V10i[Pp[s:s+2048]].reshape(-1,390).astype(np.float32)@Qf    # Qf: Q padded to 390, transposed
    ...                                                              # per-chunk top-10, merge
```

## Results

Median ms per ask, seed 202610271855 / seed 777 (p10 to p90 spread 0.03 to 2.4 ms). Recall@10 was 1.000 and top-1 128/128 for every variant.

| N | variant | bytes/row | ms/ask | vs base3 unpack f32 | vs dense int8 |
|---|---|---|---|---|---|
| 100k | dense f32gemm (int8 stored) | 384 | 1.17 / 1.13 | 1.90x / 1.89x faster | 1.00x |
| 100k | base3 unpack f32 (243x5) | 77 | 2.22 / 2.13 | 1.00x | 1.90x / 1.89x slower |
| 100k | base3 unpack int8+cast | 77 | 2.34 / 2.24 | 0.95x / 0.95x | 2.00x / 1.99x slower |
| 100k | pair decode int8+cast | 78 | **1.87 / 1.71** | **1.18x / 1.24x** | 1.60x / 1.52x slower |
| 100k | pair decode f32 | 78 | 1.96 / 1.89 | 1.14x / 1.13x | 1.67x / 1.68x slower |
| 500k | dense f32gemm (int8 stored) | 384 | 5.89 / 5.64 | 1.86x / 1.90x faster | 1.00x |
| 500k | base3 unpack f32 (243x5) | 77 | 10.96 / 10.70 | 1.00x | 1.86x / 1.90x slower |
| 500k | base3 unpack int8+cast | 77 | 11.57 / 10.93 | 0.95x / 0.98x | 1.97x / 1.94x slower |
| 500k | pair decode int8+cast | 78 | **9.15 / 8.80** | **1.20x / 1.22x** | 1.55x / 1.56x slower |
| 500k | pair decode f32 | 78 | 9.66 / 9.37 | 1.13x / 1.14x | 1.64x / 1.66x slower |

| storage | 100k | 500k |
|---|---|---|
| base-3 bytes (77 B/row) | 7,700,000 | 38,500,000 |
| pair uint16 codes (78 B/row) | 7,800,000 | 39,000,000 |
| int8 trits (384 B/row) | 38,400,000 | 192,000,000 |

## Verdict

**Not viable.** The fused two-byte int8 decode is the best packed path, but it is only 1.18x to 1.24x faster than the single-byte `unpack+gemm`, well under the 1.5x bar, at exact recall. It narrows the gap to int8 storage from about 1.9x to 1.52x to 1.60x while keeping a 4.9x memory saving (39 MB against 192 MB at 500k). The f32 pair table is slower than the int8 one (bigger gather traffic), and an int8 single-byte table plus cast is slightly slower than the f32 single-byte table, so the gain comes from halving the gather count, not from the element type. Updated `night` rule: when resident memory forces the packed index, store uint16 pair codes and use the int8 pair-table decode plus cast and `@` at 2k chunks (about 20% faster than the 14:55 rule, exact); otherwise int8 storage with cast and `@` still wins.

Caveats: synthetic embeddings only, one thread, one machine; the 590 KB pair table fits in L2 here and may not on smaller cores.

## Pair files this wave

- No pair report or Quant Chief assignment for the 18:05 wave reached this run, and the fresh push clone shows no JuniorLLM push since 15:04 MT (tip [447e548](https://github.com/cloudcover95/JuniorLLM/commit/447e548), the 14:55 bench note itself), so there is no 18:05 pair file to link and no path overlap with Trit Pack, Handshake or an LLC bot. The ask side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

The decode line is now close to exhausted for numpy. Next, check the memory side directly: under a 2 GiB MemoryMax-style budget, how many docs can each storage (int8 trits, uint16 pair codes, fp16 and f32 embeddings) hold resident with headroom for a 64-ask batch, and does the packed index let `night` cover at least 1.5x more docs per ask-second at that cap?

close: a32e6479d357db4c (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-27_1455)
