---
local: true
date: 2026-10-30
wave: "2026-10-30 18:05 (bench 18:55, ran 19:03 to 19:06)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`night` sweeping the packed ask index)"
verdict: not-viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, night, ask-index, lookup-table, fused-score, cache, bench]
---

# juniorctl night: fused per-query lookup-table scoring of uint16 pair codes vs int8 cast-and-matmul, uncapped

## Question

The 14:55 note ([[2026-10-30_1455_night-read-rate-threads-pair-codes-vs-int8]]) found that on fast storage the night sweep is bound by per-chunk compute, not bytes. Does fusing the decode into the score help? Instead of building an int8 or float32 row, score each uint16 pair code with a per-query lookup table of partial dot products. Does that beat the int8 cast-and-matmul path when uncapped at 2, 4 and 8 threads, and at what query batch size does the table stop fitting in cache?

## Method

Same synthetic index and encoding as the 02:55 to 14:55 runs: 384-dim rows ternarized with the 0.5 mean|x| absmean rule, stored as int8 trits (384 B/row) and as uint16 pair codes (39 codes, 78 B/row, each code is 10 trits). This run holds 500,000 rows in memory (192,000,000 B int8, 39,000,000 B pair) with no device model or disk reads, so it measures compute only. 64 queries, top-10, 8,192-row chunks over a thread pool, OpenBLAS pinned to 1 thread. Queries are processed in batches of B = 1, 8 or 64; one sweep scores all 64 queries.

Four scoring paths per chunk:

| variant | what it does per chunk | table per query |
|---|---|---|
| int8 | int8 rows cast to float32, then `Qb @ X.T` | none |
| decode | 59,049 x 10 int8 pair-table gather, cast, then `Qb @ X.T` (the current pair path) | none (shared 590 KB table) |
| lut10 | per query, 39 groups x 59,049 partial dots; score = sum of 39 gathers per row | 9.21 MB float32 |
| lut5 | each code split into two 5-trit halves (`code // 243`, `code % 243`); per query 78 x 243 partial dots; 78 gathers per row | 0.076 MB float32 |

lut10 at B = 64 was skipped (64 x 9.21 MB = 590 MB of tables). Each cell ran 3 repeats with the variant order reversed on odd repeats. Every sweep's top-10 was compared to an 8-thread, B = 64 int8 reference. numpy 2.5.4, 8-vCPU shared box, MemAvailable about 6.6 GB at start, python3 and numpy only, no network, nothing bound. A 65,536-row smoke run (`python3 bench.py 65536 1 1 8 1,64`) came first. Command, run in a scratch folder (not committed):

```
python3 bench.py 500000 202610301855 3 2,4,8 1,8,64     # N, seed, repeats, worker threads, query batch sizes
```

Core of the fused path (full script uses the same `tern`, `enc_pair` and pair table as the 14:55 script):

```
OFF10=(np.arange(39)*59049).astype(np.int32); OFF5=(np.arange(78)*243).astype(np.int32)
def luts(Qb,kind):
    Qp=np.zeros((len(Qb),390),np.float32); Qp[:,:D]=Qb
    if kind=="lut10": return np.einsum("bgk,vk->bgv",Qp.reshape(-1,39,10),V10f).reshape(len(Qb),-1)   # B x 39*59049
    return np.einsum("bgk,vk->bgv",Qp.reshape(-1,78,5),V5f).reshape(len(Qb),-1)                      # B x 78*243
# per chunk P (n x 39 uint16):
if kind=="lut10": idx=P.astype(np.int32)+OFF10
else: idx=np.empty((len(P),78),np.int32); idx[:,0::2]=P//243; idx[:,1::2]=P%243; idx+=OFF5
S=np.take(L,idx,axis=1).sum(2,dtype=np.float32)        # B x n scores, then argpartition top-10
```

## Results

Median of 3 full 64-query sweeps over 500,000 rows in ms, spread (max minus min) in brackets. int8/variant above 1 means the variant is faster than int8. Build took 8 s.

| batch | threads | int8 ms | decode ms | lut5 ms | lut10 ms | int8/decode | int8/lut5 | int8/lut10 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 2487 (219) | 4302 (61) | 4333 (285) | 3908 (549) | 0.58x | 0.57x | 0.64x |
| 1 | 4 | 1257 (105) | 2188 (26) | 2188 (52) | 2355 (96) | 0.57x | 0.57x | 0.53x |
| 1 | 8 | 739 (31) | 1207 (52) | 1276 (28) | 1777 (121) | 0.61x | 0.58x | 0.42x |
| 8 | 2 | 509 (52) | 708 (14) | 1700 (118) | 3200 (373) | 0.72x | 0.30x | 0.16x |
| 8 | 4 | 301 (43) | 414 (58) | 909 (104) | 2010 (143) | 0.73x | 0.33x | 0.15x |
| 8 | 8 | 163 (18) | 227 (34) | 552 (67) | 1460 (40) | 0.72x | 0.29x | 0.11x |
| 64 | 2 | 153 (3) | 183 (7) | 2703 (153) | skipped | 0.84x | 0.06x | n/a |
| 64 | 4 | 80 (6) | 97 (1) | 1519 (319) | skipped | 0.82x | 0.05x | n/a |
| 64 | 8 | 47 (8) | 55 (1) | 789 (212) | skipped | 0.86x | 0.05x | n/a |

Recall@10 against the int8 reference was 1.0000 for every variant in every cell. Fused-score error on the first chunk against a float64 dot was at most 1.89e-06 (lut10) and 2.23e-06 (lut5), so the tables are numerically fine.

No fused variant beats int8 anywhere. At B = 1, where the table advantage should be largest (no matmul batching), lut5 and lut10 are 0.42x to 0.64x of int8, about the same as the decode path. lut10's 9.2 MB per-query table misses cache badly as threads go up: it goes from 0.64x at 2 threads to 0.42x at 8 threads at B = 1, and to 0.11x at B = 8 (74 MB of tables). lut5's 76 KB table fits in L2 at every batch size, but the cost per row is 78 gathers per query, and that grows linearly with B while sgemm amortizes: by B = 64, lut5 needs 789 ms against 47 ms for int8 (0.05x). The table stops paying the moment more than one query shares a chunk. Larger batches help int8 and decode (sgemm reuse) and hurt every gather-based path.

Batch size matters far more than format: going from B = 1 to B = 64 cuts the int8 sweep 15.7x at 8 threads (739 to 47 ms). At the night path's B = 64, the existing decode path is within 1.2x of int8 (0.82x to 0.86x), so the decode is no longer what pair codes lose on.

## Verdict

**Not viable.** Fused lookup-table scoring is 0.05x to 0.64x of int8 cast-and-matmul at every batch size and thread count tested, with exact top-10 throughout. It is also slower than or equal to the existing pair-table decode in every cell except lut10 at B = 1 and 2 threads (3,908 vs 4,302 ms, 1.10x, still slower than int8). Keep the decode-then-sgemm path for `night`, and always batch all queries per chunk: B = 64 is 15.7x faster than B = 1 for int8 and 21.9x for decode at 8 threads. The pair-code win from 14:55 (2.22x at about 400 MB/s or slower) still comes only from bytes read, not compute.

Caveats: numpy `take` gathers stand in for what a C or SIMD kernel (shuffle-based lookup, like BitNet's TL kernels) would do, so this says nothing about a native fused kernel, only that it cannot be done in numpy alone. 500,000 rows in memory, synthetic embeddings, one seed, 3 repeats, one shared 8-vCPU box.

## Pair files this wave

- No pair file was produced for the 18:05 wave: Quant Chief gave "no gap" to both pairs (thirty-fourth idle wave, roadmap blobs unchanged). In JuniorLLM, the only commit in the last 6 hours is the 14:55 note itself ([e7e21a7](https://github.com/cloudcover95/JuniorLLM/commit/e7e21a7)), so no Trit Pack, Handshake or LLC path was touched. This note adds one new file under `junior_osai/vault/findings/` only. The ask/night side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Since batching dominates, does the night sweep gain more from a larger chunk with all 64 queries per chunk than from the storage format? At B = 64 and 8 threads, how do 2,048, 8,192, 32,768 and 131,072-row chunks change int8 and decode sweep time and peak float32 working memory, and is there a chunk size where decode matches int8 within 1.05x?

close: 888a8eb9a0884dfc (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-30_1455)
