---
local: true
date: 2026-10-27
wave: "2026-10-27 02:05 (bench 02:55, ran 03:00)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl ask` / `night` / `security` (batched `night` sweep over a resident ternary ask index)"
verdict: not-viable
tags: [osai, vault, findings, ternary, base3, i2s, retrieval, juniorctl, night, batch, lut, gather, bench]
---

# juniorctl night: batch-size sweep for the shared table, 8-bit (2-bit pack) vs base-3 single-byte 5-trit

## Question

The 18:55 note ([[2026-10-26_1855_ask-batch16-shared-lut-base3-vs-8bit]]) found that one 8-bit table shared by 16 asks runs 2.90x to 3.06x faster per ask than today's single-ask scan, while the big base-3 pair table got slower when shared. Its next question: where is the batch-size sweet spot for the shared table (4 to 128 asks), and does the same batching on the base-3 single-byte 5-trit table (77 x 243 x B, 77 B/row, 20% smaller index) beat the 8-bit table at the same batch size by the 1.5x bar? Quant Chief gave both pairs "no gap" at 02:05 (fifteenth idle wave), so this run stays on the ask/night line.

## Method

Same synthetic clustered unit embeddings as the 10:55, 14:55 and 18:55 runs: 384 dims, 256 topic centers plus noise, 20,000 and 100,000 docs, absmean-threshold ternarization (|x| > 0.5 mean|x|). 128 noisy normalized queries per size. Each batch size B in {4, 8, 16, 32, 64, 128} runs the 128 asks as 128/B batches, three passes (six for B=128); per-ask time is batch time divided by B, and the table build is inside the timed path. Baseline is today's single-ask 8-bit kernel over 64 asks. Error is max |score - exact ternary dot| over all 128 asks; top-1 agreement is against the exact ternary scores. numpy 2.5.4, one thread, Intel Xeon (16 MiB L2 across 8 instances, 320 MiB L3), two seeds (202610270255 and 777), python3 and numpy only, no network.

Command, run in a scratch folder (not committed):

```
OPENBLAS_NUM_THREADS=1 python3 bench.py          # seed 202610270255
OPENBLAS_NUM_THREADS=1 python3 bench.py 777
```

Core kernels (data generator is the 18:55 script unchanged; `scan` is the same gather-and-add loop):

```python
def b8(I,Q):   # I: 96 x N uint8 (2-bit pack, transposed); table 96 x 256 x B
    B=Q.shape[0]; Tab=np.ascontiguousarray((Q.reshape(B,96,4)@V4.T).transpose(1,2,0)); return scan(I,Tab)
def b5(I,Q):   # I: 77 x N uint8 (5 trits per byte, base 3); table 77 x 243 x B
    B=Q.shape[0]; Qp=np.concatenate([Q,np.zeros((B,1),np.float32)],1).reshape(B,77,5)
    Tab=np.ascontiguousarray((Qp@V5.T).transpose(1,2,0)); return scan(I,Tab)
```

## Results

Median ms per ask (p10 to p90 spread in brackets). The last column is base-3 5-trit against 8-bit at the same batch size, which is the head-to-head this note asks about.

Seed 202610270255 (8-bit single ask: 5.21 ms at 20k, 23.1 ms at 100k):

| B | table KiB (8-bit / base-3) | 20k 8-bit | 20k base-3 | 20k base-3 vs 8-bit | 100k 8-bit | 100k base-3 | 100k base-3 vs 8-bit |
|---|---|---|---|---|---|---|---|
| 4 | 384 / 292 | 5.05 (0.71) | 4.05 (0.35) | 1.25x | 27.3 (4.5) | 20.1 (1.6) | 1.36x |
| 8 | 768 / 585 | 2.45 (0.18) | 2.01 (0.20) | 1.22x | 14.8 (1.7) | 11.5 (0.53) | 1.29x |
| 16 | 1,536 / 1,169 | 1.40 (0.13) | 1.20 (0.17) | 1.17x | 7.98 (0.90) | 6.61 (1.7) | 1.21x |
| 32 | 3,072 / 2,339 | 1.22 (0.08) | 0.96 (0.12) | 1.28x | 6.11 (0.71) | 4.72 (0.27) | 1.29x |
| 64 | 6,144 / 4,678 | 1.04 (0.11) | **0.81 (0.11)** | 1.28x | 5.07 (0.20) | **3.93 (0.74)** | 1.29x |
| 128 | 12,288 / 9,356 | 0.96 (0.15) | 0.81 (0.04) | 1.19x | 7.39 (0.88) | 5.48 (0.83) | 1.35x |

Seed 777 (8-bit single ask: 4.54 ms at 20k, 22.2 ms at 100k):

| B | 20k 8-bit | 20k base-3 | 20k base-3 vs 8-bit | 100k 8-bit | 100k base-3 | 100k base-3 vs 8-bit |
|---|---|---|---|---|---|---|
| 4 | 5.18 (1.8) | 4.01 (0.56) | 1.29x | 26.3 (5.4) | 19.9 (1.7) | 1.33x |
| 8 | 2.57 (0.19) | 2.12 (0.18) | 1.21x | 14.0 (2.4) | 10.9 (0.97) | 1.29x |
| 16 | 1.52 (0.18) | 1.17 (0.07) | 1.30x | 8.06 (0.37) | 6.31 (0.24) | 1.28x |
| 32 | 1.18 (0.15) | 0.92 (0.06) | 1.28x | 5.79 (0.39) | 4.62 (0.20) | 1.25x |
| 64 | 0.97 (0.07) | **0.77 (0.03)** | 1.25x | 4.79 (0.14) | **3.95 (0.08)** | 1.21x |
| 128 | 0.92 (0.07) | 0.77 (0.04) | 1.19x | 6.43 (0.99) | 5.33 (0.69) | 1.21x |

| layout | index bytes at 100k | B/row | max abs error | top-1 agreement |
|---|---|---|---|---|
| 2-bit pack, 8-bit table | 9,600,000 | 96 | 3.3e-06 to 3.8e-06 | 128/128 |
| base-3 5 trits per byte, 243-entry table | 7,700,000 | 77 | 3.3e-06 to 4.5e-06 | 128/128 |

## Verdict

**Not viable for the head-to-head.** At every batch size the base-3 5-trit table is only 1.17x to 1.36x faster than the 8-bit table at the same batch (1.21x to 1.29x at the B=64 sweet spot), and its index is 1.25x smaller (77 vs 96 B/row). Both are real, consistent wins with identical scores, but neither clears the 1.5x bar, so base-3 over 2-bit stays an incremental layout choice, not a ternary breakthrough.

Side findings worth keeping for `juniorctl night`:

- The batch sweet spot is B=64 at both sizes for both layouts. Gains flatten at 20k past 64 and reverse at 100k at B=128 (the N x 128 f32 accumulator is 51 MB and falls out of L2), so B=64 is the cap.
- Batch 4 is no help, or a loss, for the 8-bit table at 100k (0.84x to 0.85x of single ask); batching only pays from B=8 up.
- Base-3 5-trit at B=64 is 1.72x to 1.96x faster per ask than the 18:55 rule (8-bit at B=16) at 20k and 2.03x to 2.04x at 100k, and 5.6x to 6.4x faster than today's single-ask scan. Most of that gain comes from the larger batch, not from base-3, so it doesn't count as a ternary result. Synthetic embeddings only.
- Updated rule: `night` sweeps should use the base-3 5-trit index with a 64-ask shared table (4.6 MiB); single asks keep the base-3 pair table above about 50k rows.

## Pair files this wave

- No pair report reached this run, and Quant Chief's 02:05 assignment was "no gap" for both pairs (fifteenth idle wave), so there is no new pair file for the 02:05 wave. The Chief left JuniorHome [5ae1679](https://github.com/cloudcover95/JuniorHome/commit/5ae1679) (web3node/advance_gguf_probe_future.py) to the LLC. JuniorLLM's tip at clone time was [53aa28b](https://github.com/cloudcover95/JuniorLLM/commit/53aa28b) (02:52 MT, cloudcover95, "rails/lanes: offline lane lock check"), which no roadmap assignment covers this wave. It sits outside the findings folder, so there's no path overlap. The ask side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

With B=64 fixed, does a top-k step fused into the scan (an argpartition per 8k-row chunk instead of holding the full N x 64 accumulator) keep the base-3 5-trit night sweep flat at 100k to 500k docs, and does int16 accumulation of a scaled table (half the accumulator bytes) clear 1.5x over the f32 accumulator at acceptable top-1 agreement?

close: 832f334b7ebbda20 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-26_1855)
