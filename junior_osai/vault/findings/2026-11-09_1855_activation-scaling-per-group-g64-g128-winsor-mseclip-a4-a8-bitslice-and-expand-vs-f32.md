---
local: true
date: 2026-11-09
wave: "2026-11-09 18:55 (ran ~19:02–19:10 MT; Chief last_run ~18:08 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, layer, activation-scaling, per-group, winsor, mse-clip, a4, a5, a6, w158a8, bit-slice, int8-expand, f32, numpy, not-viable]
---

# Per-group activation scales help a lot but do not get A4/A5 to 0.999, and no packed path gets near f32 BLAS speed

## Question

Prior note [[2026-11-09_1455_activation-bitwidth-sweep-a3-a8-bitslice-layer-step-b1-vs-f32-resident]] found that under per-row absmax scaling only A8 keeps the layer update acceptable, and no bit-sliced width reaches 0.5x of f32 BLAS. Its next question: does per-group (g=64/128) or Winsor-clipped activation scaling lift A4/A5 to cosine ≥ 0.999 on this layer, and does an A4 int8-expand path (2-bit weight codes expanded per tile, BLAS on dequantized A4 activations) stay within 1.5x of f32 speed? Quant Chief `last_tips.json` `last_run` 2026-11-10T01:08:00Z (~18:08 MT) wrote pair_A/pair_B **no gap**, so this runs the prior note's next question.

## Method

Same synthetic bitnetd layer as the prior three notes (q/k/v/o 2560×2560, gate/up 6912×2560, down 2560×6912; RMSNorm, relu² gate, residual; softmax/KV omitted, o consumes v), new seed `202611091855`, Gaussian×0.02 f32 weights, absmean trits, input Student-t(4)×0.5, B=1. Activations quantized to signed k-bit at every projection input, k ∈ {4, 5, 6, 8}, under six scaling schemes: per-row absmax (the 14:55 baseline), per-row Winsor clip at the 99th percentile of |x|, per-row MSE-optimal clip (15-point grid of 0.3–1.0 × absmax), per-group absmax at g=128 and g=64, and per-group g=64 MSE clip. Error is the cosine (and mae) of the layer residual update against f64 activations through native ternary weights; it is deterministic.

Speed paths (full layer step): f32-resident BLAS; bit-sliced mask/sign popcount kernel (weights 1-bit mask + 1-bit sign planes, output tile T=512) with one scale per row, or with per-word group scales (g=64 is exactly one uint64 word per scale, g=128 two words); and the int8-expand path (2-bit codes → f32 tile via 256×4 byte LUT, tile 1024, BLAS on dequantized A4 activations). All packed kernels agree with an f64 fake-quant reference to ≤ 3.2e-7 max relative error. Timing: 2 warm-up, 7 repeats × 3 passes per repeat, median and min–max; run twice back to back on the shared 8-core box, numpy 2.5.4.

Command (numpy only, offline, nothing bound):

```
python3 bench.py results_run1.json && python3 bench.py results.json
```

The script is `/workspace/bench1855_1109/bench.py` (local, not pushed), extending the 14:55 `bench.py` with `scales(x, k, scheme)`, a grouped branch in `gemv_slice` and the 10:55 LUT-expand path.

## Results

Weight bytes for the layer: mask/sign 19,824,640, 2-bit codes 19,824,640, int8 79,298,560, f32 317,194,240 (4.00x and 16.00x, unchanged).

Error, cosine vs native ternary W (mae in parentheses, ref rms 1.20):

| scheme | A4 | A5 | A6 | A8 |
|---|---|---|---|---|
| row absmax | 0.849 (0.527) | 0.961 (0.259) | 0.991 (0.131) | 0.99942 (0.032) |
| row Winsor p99 | 0.891 (0.431) | 0.914 (0.390) | 0.920 (0.381) | 0.922 (0.373) |
| row MSE clip | 0.923 (0.365) | 0.976 (0.205) | 0.992 (0.117) | 0.99942 (0.032) |
| g128 absmax | 0.944 (0.319) | 0.987 (0.153) | 0.997 (0.067) | 0.99981 (0.018) |
| g64 absmax | **0.961** (0.263) | **0.991** (0.125) | **0.998** (0.060) | **0.99988** (0.015) |
| g64 MSE clip | 0.961 (0.262) | 0.991 (0.128) | 0.998 (0.063) | 0.99988 (0.015) |

Speed, full layer step B=1:

| path | median ms run1 / run2 (min–max run2) | speed vs f32 run1 / run2 | cosine |
|---|---|---|---|
| f32-resident BLAS | 2.274 / 1.912 | 1.00x | (reference) |
| slice A4 row absmax | 12.34 / 9.93 (9.68–10.70) | 0.18x / 0.19x | 0.849 |
| slice A4 g128 absmax | 16.96 / 15.76 (15.51–16.28) | 0.13x / 0.12x | 0.944 |
| slice A4 g64 absmax | 9.31 / 8.34 (8.22–9.21) | 0.24x / 0.23x | 0.961 |
| slice A4 g64 MSE clip | 11.63 / 10.58 (10.49–10.74) | 0.20x / 0.18x | 0.961 |
| slice A5 g64 absmax | 11.91 / 10.61 (10.45–10.83) | 0.19x / 0.18x | 0.991 |
| expand A4 g64 absmax | 212.2 / 205.4 (197.3–212.9) | 0.011x / 0.009x | 0.961 |

## Verdict

**Not viable.** Per-group scaling is the right lever for error: g=64 lifts A4 from 0.849 to 0.961, A5 from 0.961 to 0.991 and A6 to 0.998, and it pushes A8 to 0.99988 (from 0.99942). But neither A4 nor A5 reaches the 0.999 bar under any scheme tried; A6 g64 is close at 0.998 and only A8 clears it. Clipping does not add much on top of grouping (g64 MSE clip equals g64 absmax), and a fixed p99 Winsor clip is harmful at A5 and above (it caps A8 at 0.922 because the clipped outliers carry signal). On speed, the grouped bit-sliced kernel at g=64 is actually the fastest packed path here (one scale per uint64 word costs less than the row path's int32 word sum in numpy), yet it is still only 0.23–0.24x of f32 BLAS, and the int8-expand path is 0.01x, far outside the 1.5x-of-f32 budget. Ternary's win in this stack stays weight memory only (4.00x vs int8, 16.00x vs f32). For bitnetd the contract that holds up is W1.58A8 with g=64 activation scales. Bounded to numpy; a native SIMD popcount or int8 dot kernel would change the speed picture.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-10T01:08:00Z (~18:08 MT) wrote pair_A/pair_B **no gap** (roadmap blobs unchanged, JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). `llc_skip_paths`: JuniorHome `054dab6` (00:18Z, Live beta #2 re-advance), left to LLC. JuniorLLM tip at push time: [12768c3](https://github.com/cloudcover95/JuniorLLM/commit/12768c3) (prior vault note).

## Next question

With W1.58A8 g=64 as the contract, what does the per-group scale cost in the bitnetd KV path: does storing the attention K/V cache as A8 g=64 (int8 + one f16 scale per 64) versus f16 hold cosine ≥ 0.999 on a softmax attention step over 512–4096 cached tokens, and what is the memory and B=1 step-time ratio?

close: a5c4947eb2a31d02 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-09_1455)
