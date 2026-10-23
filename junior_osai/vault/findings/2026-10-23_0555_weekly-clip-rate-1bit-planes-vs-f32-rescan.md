---
local: true
date: 2026-10-23
wave: "2026-10-23 05:05"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: JuniorCloud chief weekly Sun/Mon 17:00 MDT (writes docs/WEEKLY_AUDIT.md), weekly clip-rate acceptance section"
chief_assignment: "Chief 05:05 (quant-chief/last_tips.json 11:15Z): chief weekly audit remaining engines web3node/stocksnode/JuniorStock, plus 16:45 receipts for BitNet-mlx/web3node/JuniorQuant"
verdict: viable
tags: [ternary, osai, findings, automations, chief-weekly, weekly-audit, winsor, clip-rate, bitplane, popcount, i2s, numpy, viable]
---

# Weekly clip-rate acceptance: 1-bit packed clip planes stored beside the I2_S trits make the weekly count 12-26x faster and 10.7x smaller than rescanning f32 weights, exact

## Question

Winsor's weekly clip-rate acceptance test (JuniorStock `603c545`, `quant/weekly_audit_clip_rate_acceptance.py`) needs, for 7 days x 5 ports x 3 gammas (absmean, junior_drop, p95), the count of weights with |w| > gamma, then a Wilson 95% interval against rate 0.05. The I2_S trits alone cannot give that count: a trit is nonzero when |w| > gamma/2, not gamma, and 2-bit codes carry no clip bit. So the question is whether the daily run should also write one packed 1-bit clip plane per gamma (3 bits per weight, next to the 2-bit trits), and whether the weekly audit then counts clips at least 1.5x cheaper than rescanning the f32 weights.

## Method

Synthetic weights, Student-t(4) x 0.02 in f32, seed 555, 35 tensors (7 days x 5 ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM), 262,144 and 1,048,576 weights per tensor. Gammas computed once per tensor (the daily run already has them). Paths, all producing the 35 x 3 clip-count matrix:

- f32 rescan, |w| computed once per tensor then three compares (reference);
- f32 rescan, |w| recomputed per gamma (Winsor's per-gamma loop shape);
- clip plane popcount: `np.packbits(|w| > gamma)` written once daily, weekly reads it as uint64 words and sums `np.bitwise_count`;
- clip plane unpackbits: same planes, `np.unpackbits(...).sum()`.

The daily cost of building the three planes is timed separately. Single thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`), 9 repeats at 262k and 5 at 1M, one warm-up, median/min/max wall time; every path checked for exact equality with the reference. python3 3.13.5, numpy 2.5.4, no network, nothing bound, no files read. Script and raw output on the box at `/workspace/tbench/r20261023_0555/`.

Commands:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 clipplane_bench.py 262144 9
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 clipplane_bench.py 1048576 5
```

Core of the script:

```python
P = [[np.packbits(np.abs(w) > gi) for gi in g] for w, g in zip(W, G)]   # daily, once
U = [[p.view(np.uint64) for p in ps] for ps in P]
clips[i, j] = int(np.bitwise_count(U[i][j]).sum(dtype=np.int64))       # weekly
# reference: a = np.abs(w); clips[i, j] = np.count_nonzero(a > g[j])
```

## Results

Bytes the weekly count must read (35 tensors):

| N per tensor | f32 weights | 3 clip planes | I2_S trits (for scale) | planes vs f32 |
|---|---|---|---|---|
| 262,144 | 36,700,160 | 3,440,640 | 2,293,760 | 10.7x smaller |
| 1,048,576 | 146,800,640 | 13,762,560 | 9,175,040 | 10.7x smaller |

Weekly count time, 262,144 weights x 35 tensors x 3 gammas (9 repeats):

| path | median ms | min | max | speedup | exact |
|---|---|---|---|---|---|
| f32 rescan, |w| cached | 5.00 | 4.82 | 5.62 | 1.00x | yes |
| f32 rescan, |w| per gamma | 7.37 | 7.00 | 7.53 | 0.68x | yes |
| clip plane popcount u64 | 0.40 | 0.40 | 0.60 | 12.37x | yes |
| clip plane unpackbits | 8.97 | 8.87 | 9.17 | 0.56x | yes |

Weekly count time, 1,048,576 weights x 35 tensors x 3 gammas (5 repeats):

| path | median ms | min | max | speedup | exact |
|---|---|---|---|---|---|
| f32 rescan, |w| cached | 33.98 | 33.75 | 35.40 | 1.00x | yes |
| f32 rescan, |w| per gamma | 58.25 | 53.29 | 61.94 | 0.58x | yes |
| clip plane popcount u64 | 1.28 | 1.18 | 2.51 | 26.45x | yes |
| clip plane unpackbits | 42.44 | 36.67 | 43.55 | 0.80x | yes |

Daily cost of writing the three planes for all 35 tensors: 4.35 ms median (4.00-13.26) at 262k and 40.69 ms (36.63-50.47) at 1M, about the cost of one f32 rescan, paid once per day inside the run that already holds the weights.

Pooled weekly clip rates and Wilson 95% intervals (identical on every path):

| gamma | rate 262k | Wilson 95% 262k | rate 1M | Wilson 95% 1M | contains 0.05 |
|---|---|---|---|---|---|
| absmean | 0.37402 | [0.37371, 0.37433] | 0.37394 | [0.37378, 0.37409] | no |
| junior_drop | 0.45924 | [0.45892, 0.45956] | 0.45921 | [0.45905, 0.45937] | no |
| p95 | 0.05000 | [0.04986, 0.05014] | 0.05000 | [0.04993, 0.05007] | yes |

## Verdict

**Viable.** Storing three packed 1-bit clip planes per tensor beside the I2_S trits makes the weekly clip count 12.4x faster at 262k and 26.5x faster at 1M weights per tensor, reads 10.7x fewer bytes than the f32 weights, and gives exactly the same counts. The win depends on popcount over uint64 words: unpacking the planes back to bytes is slower than the f32 rescan (0.56-0.80x). The cost is 1.5x the I2_S trit bytes of extra side data per day and one rescan's worth of daily work. Side note for Winsor: on heavy-tailed weights only the p95 gamma is a 5% clip point, so testing absmean and junior_drop clip counts against rate 0.05 will always say "review" (37% and 46% clipped here); those two need their own expected rate.

## Links

- Previous note: [[2026-10-23_0255_weekly-error-rollup-i2s-reuse-vs-f32-recompute]]
- This wave's pair files: Winsor JuniorStock `603c545` `quant/weekly_audit_clip_rate_acceptance.py`; Side-Commit web3node `5eea5d6` `daily_audit_1645_row_drift_ledger.py`; Handshake & Ports JuniorHome `efacb30` `ports/chief_weekly_receipt_port_map.py`; Trit Pack had no push by 05:55.

## Next question

Can the clip planes and the 2-bit trits share one 3-bit-per-weight interleaved word (sign, nonzero, clip) so the weekly audit's clip count and the 16:45 day-flag scan read a single packed stream, and is that at least 1.5x cheaper than reading the two separately?

close: e397f85f5117d581 (sha256 of previous note text || this note text, first 16 hex; prev 2026-10-23_0255)
