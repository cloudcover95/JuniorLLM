---
local: true
date: 2026-10-26
wave: "2026-10-26 18:05 (bench 18:55, ran 19:05)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl ask` / `night` / `security` (batched `night` sweep over a resident ternary ask index)"
verdict: not-viable
tags: [osai, vault, findings, ternary, base3, i2s, retrieval, juniorctl, ask, night, batch, lut, gather, bench]
---

# juniorctl night: sharing one lookup table across a 16-ask batch, base-3 pair table vs the 8-bit table

## Question

The 14:55 note ([[2026-10-26_1455_ask-index-base3-10trit-lut-vs-2bit-8bit-16bit]]) found the base-3 uint16-pair layout (10 trits per uint16, 77 B/row) scans 2.11x to 2.16x faster than the 8-bit table at 100k docs, but at 20k its 9 MB per-ask table build kept it under 1.5x. Its next question: if a `juniorctl night` sweep asks 16 questions at once, can one 38 x 59,049 x 16 table be built and gathered 16 floats at a time, so the per-ask cost at 20k drops below 2.8 ms and clears 1.5x over the 8-bit scan? For a fair answer the 8-bit table gets the same batching (96 x 256 x 16). Quant Chief gave both pairs "no gap" at 18:05, so this run stays on the ask line.

## Method

Same synthetic clustered unit embeddings as the 10:55 and 14:55 runs: 384 dims, 256 topic centers plus noise, 20,000 and 100,000 docs, absmean-threshold ternarization (|x| > 0.5 mean|x|). 96 noisy normalized queries per size, run as 96 single asks or as 6 batches of 16. Every table is built from the queries inside the timed path. Per-ask time is the batch time divided by 16. Error is max |score - exact ternary dot| over all 96 asks; top-1 agreement is against the exact ternary scores. numpy 2.5.4, one thread, two seeds (202610261855 and 777), python3 and numpy only, no network.

Layouts:

1. fp32 BLAS, `X @ Q.T` per batch of 16 (reference only; it scores the float vectors, not the trits).
2. 2-bit pack, 8-bit table, single ask (96 gathers, 96 B/row), today's kernel.
3. Base-3 uint16 pair, 10-trit table, single ask (39 gathers, 77 B/row), the 14:55 winner.
4. 2-bit pack, 8-bit table shared by 16 asks: 96 x 256 x 16 f32 (1.5 MiB), each gather pulls a 16-float row.
5. Base-3 uint16 pair, 10-trit table shared by 16 asks: 38 x 59,049 x 16 f32 (137 MiB), each gather pulls a 16-float row.

Command, run in a scratch folder (not committed):

```
OPENBLAS_NUM_THREADS=1 python3 bench.py          # seed 202610261855
OPENBLAS_NUM_THREADS=1 python3 bench.py 777
```

Core of the script (the data generator and single-ask kernels are the 14:55 script unchanged):

```python
NB=16
def b8(I,Q):                       # I: 96 x N uint8 (2-bit pack, transposed); Q: 16 x 384
    Tab=np.ascontiguousarray((Q.reshape(NB,96,4)@V4.T).transpose(1,2,0))   # 96 x 256 x 16
    s=Tab[0][I[0]].copy()
    for j in range(1,96): s+=Tab[j][I[j]]                                    # N x 16
    return s
def b10(I,Q):                      # I: (38 x N uint16 base-3 pairs, N uint8 last byte)
    BT,last=I; Qp=np.concatenate([Q,np.zeros((NB,1),np.float32)],1).reshape(NB,77,5)
    T=np.ascontiguousarray((Qp@V5.T).transpose(1,2,0))                       # 77 x 243 x 16
    T2=T[:76].reshape(38,2,243,NB)
    Tab=(T2[:,0][:,None,:,:]+T2[:,1][:,:,None,:]).reshape(38,59049,NB)       # 137 MiB
    s=Tab[0][BT[0]].copy()
    for j in range(1,38): s+=Tab[j][BT[j]]
    s+=T[76][last]; return s
```

## Results

Median ms per ask (p10 to p90 spread in brackets), speedup against the single-ask 8-bit scan.

| seed, docs | fp32 BLAS batch 16 | 8-bit single | base-3 pair single | 8-bit batch 16 | base-3 pair batch 16 |
|---|---|---|---|---|---|
| 202610261855, 20k | 0.22 | 4.76 (0.49), 1.00x | 3.29 (0.40), 1.44x | **1.64 (0.70), 2.90x** | 6.05 (0.47), 0.79x |
| 777, 20k | 0.22 | 4.77 (0.45), 1.00x | 3.31 (0.36), 1.44x | **1.56 (0.05), 3.06x** | 6.90 (0.94), 0.69x |
| 202610261855, 100k | 1.74 | 23.0 (1.9), 1.00x | 10.97 (2.2), 2.10x | **10.32 (6.8), 2.23x** | 17.16 (2.3), 1.34x |
| 777, 100k | 1.89 | 23.0 (1.3), 1.00x | 11.88 (2.3), 1.94x | **8.88 (0.82), 2.59x** | 14.23 (4.1), 1.62x |

| layout | index bytes at 100k | B/row | shared table | max abs error | top-1 agreement |
|---|---|---|---|---|---|
| fp32 BLAS | 153,600,000 | 1,536 | none | n/a (float scores) | 24 to 25 of 96 vs ternary |
| 8-bit, single or batch 16 | 9,600,000 | 96 | 96 KiB single, 1.5 MiB batch | 3.6e-06 to 4.1e-06 | 96/96 |
| base-3 pair, single or batch 16 | 7,700,000 | 77 | 9 MiB single, 137 MiB batch | 3.1e-06 to 3.3e-06 | 96/96 |

Base-3 batch-16 breakdown per batch: table build 51.5 to 66.6 ms (the same at both sizes) and scan 43.8 ms at 20k, 165 to 175 ms at 100k.

## Verdict

**Not viable for the question asked.** Sharing the base-3 pair table across 16 asks makes it slower, not faster: the shared table grows to 137 MiB, its build alone takes 52 to 67 ms per batch (3.3 to 4.2 ms per ask), and every 64-byte gathered row misses cache. At 20k it falls to 0.69x to 0.79x of the single-ask 8-bit scan, worse than the base-3 single ask it was meant to rescue.

Side finding worth keeping: batching the plain 8-bit table works. Its shared table is only 1.5 MiB, so it stays in cache, and each gather does 16 asks of work. That gives 2.90x to 3.06x over today's single-ask 8-bit scan at 20k (1.56 to 1.64 ms per ask, under the 2.8 ms target) and 2.23x to 2.59x at 100k, with identical scores on the same 96 B/row index. It also beats the base-3 single ask by 2.0x at 20k. It still loses to batched fp32 BLAS on raw speed (about 7x at 20k, 5x at 100k), so ternary here remains a memory win (16x to 20x smaller index) made much cheaper, not a speed win over float. Synthetic embeddings only. The rule for `juniorctl night`: batch the 8-bit table; keep base-3 pairs for single asks above about 50k rows.

## Pair files this wave

- No pair report reached this run, and Quant Chief's 18:05 assignment was "no gap" for both pairs (fourteenth idle wave), so there is no new pair file for the 18:05 wave. JuniorLLM's tip at clone time was still the previous findings note [2e2d41e](https://github.com/cloudcover95/JuniorLLM/commit/2e2d41e). The ask side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Where is the batch-size sweet spot for the 8-bit shared table (4, 8, 16, 32, 64 asks; the table grows 96 KiB per ask and should leave L2 somewhere past 16), and does the same batching on the base-3 single-byte 5-trit table (77 x 243 x B, 77 B/row) beat it while keeping the 20% smaller index?

close: e3ece7c75712153f (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-26_1455)
