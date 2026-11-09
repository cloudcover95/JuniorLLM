---
local: true
date: 2026-11-09
wave: "2026-11-09 10:55 (ran ~11:05–11:15 MT; Chief last_run ~10:07 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, layer, w158a8, int8-activation, bit-slice, popcount, lut, f32, numpy, not-viable]
---

# W1.58A8 fixes the error (cosine 0.9995) but on numpy both packed-weight kernels are 10x or more slower than f32-resident BLAS at B=1

## Question

Prior note [[2026-11-09_0555_bitnetd-layer-decode-step-e2e-packed-ternary-vs-int8-expand-f32-amdahl-and-error]] showed ternary activations wreck the layer update (cosine 0.44) and asked: at B=1 decode, does ternary-weight × **int8-activation** (W1.58A8, the real bitnetd BitNet b1.58 contract) via packed weights get within 1.5x of f32-resident BLAS speed while keeping the 4x / 16x memory win and cosine ≥ 0.999 on the layer update? Quant Chief `last_tips.json` `last_run` 2026-11-09T17:07:00Z (~10:07 MT) wrote pair_A/pair_B **no gap**, so this runs the prior note's next question.

## Method

Same synthetic bitnetd layer shape as the prior note (q/k/v/o 2560×2560, gate/up 6912×2560, down 2560×6912), seed `202611091055`, Gaussian×0.02 f32 weights, absmean trits; input Student-t(4)×0.5, B=1. Activations quantized per row absmax to int8 at every projection input. Two packed-weight kernels:

- **a8slice**: weights as mask/sign 1-bit planes; int8 activation split into 7 magnitude bit-planes plus a sign plane; dot = Σ_b 2^b (popcount(M_b & W_m) − 2·popcount(M_b & W_m & (S_x ^ W_s))), output-tiled T∈{128, 512}. Checked integer-exact against an int64 `q @ trits.T` reference (**exact: True**).
- **lutexp**: weights as 2-bit codes (4 per byte); each output tile T∈{256, 1024} is decoded through a 256×4 f32 byte LUT into a cache-sized f32 tile, then BLAS on the int8-dequantized activations.

Baseline: f32 weights resident (BLAS, 16x the packed bytes). Glue identical in every path (RMSNorm, o consumes v with softmax/KV left out, residual, relu²(gate)·up, sub-RMSNorm, down, residual). Error on the layer's residual update (out − x) against an f64 layer with the native ternary weights and f64 activations (isolates the int8-activation error) and against f64 with the original f32 weights. 3 warm + 7 timed repeats of 4 passes each; whole script run **twice**. numpy 2.5.4, 8 cores, offline, nothing bound. Command (scratch `/workspace/bench1055_1109/`, inline script `bench.py`):

```
python3 bench.py results_run1.json > run1.txt && python3 bench.py results.json > run2.txt
```

## Results

Weight bytes per layer: mask/sign **19 824 640**, 2-bit codes 19 824 640, int8 79 298 560 (4.00x), f32 317 194 240 (16.00x).

| path (best T) | layer ms med (min–max), run1 | run2 | speed vs f32-resident run1 / run2 (worst) |
|---|---|---|---|
| f32-resident BLAS | 2.684 (2.443–4.912) | 2.405 (2.193–3.098) | 1.00x |
| a8slice T=512 | 24.46 (22.08–27.39) | 24.24 (24.02–24.95) | **0.110x / 0.099x** (0.089x / 0.088x) |
| a8slice T=128 | 24.94 (23.66–27.50) | 25.39 (24.69–25.67) | 0.108x / 0.095x |
| lutexp T=256 | 217.7 (208.4–230.1) | 223.6 (216.0–236.3) | **0.012x / 0.011x** (0.011x / 0.009x) |
| lutexp T=1024 | 260.7 (246.7–277.1) | 266.2 (246.2–280.1) | 0.010x / 0.009x |

Layer residual-update error (identical in both runs, deterministic):

| path | reference | mae | ref rms | cosine |
|---|---|---|---|---|
| a8slice / lutexp (W1.58A8) | f64, native ternary weights | 0.0302 | 1.160 | **0.99948** |
| a8slice / lutexp (W1.58A8) | f64, original f32 weights | – | – | 0.508 |
| native ternary W, f64 act | f64, original f32 weights | – | – | 0.508 |

For comparison, the prior note's ternary-activation packed×packed layer was 4.3 ms at cosine 0.44 vs native ternary weights.

## Verdict

**Not viable on speed; error is now acceptable.** Moving to int8 activations brings the layer update to cosine **0.9995** against native ternary weights (from 0.44), so W1.58A8 is the right contract. But the speed it costs is large: the bit-sliced popcount kernel needs 7 magnitude planes (14 popcounts per word instead of 2), so the layer takes ~24 ms, **~10x slower than f32-resident BLAS** (0.10–0.11x) and ~5.7x slower than the ternary-activation kernel. The LUT-decode-to-f32 tile path is far worse (~0.011x), because numpy fancy-index decode costs far more than the BLAS it feeds. Neither comes within 1.5x of f32. Ternary's real win stays **memory only (4.00x vs int8, 16.00x vs f32)**, which matters when f32 or int8 weights don't fit, not for speed on numpy. The 0.508 cosine against the original f32 weights is the synthetic Gaussian weights' ternarization error, not the kernel's, and a trained ternary model would not pay it. Bounded to numpy `bitwise_count` and BLAS on a shared 8-core box; a native SIMD int8×trit kernel (as in bitnet.cpp) would behave differently.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-09T17:07:00Z (~10:07 MT) wrote pair_A/pair_B **no gap** (roadmap blobs unchanged, JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). `llc_skip_paths`: JuniorHome ROADMAP advance (tip `47a0cbf`, LLC). JuniorLLM tip at push time: [6521506](https://github.com/cloudcover95/JuniorLLM/commit/6521506) (prior vault note).

## Next question

At B=1, what is the smallest activation bit-width (A4/A5/A6, i.e. 3–5 magnitude planes in the bit-sliced kernel) that keeps the layer update at cosine ≥ 0.999 against native ternary weights, and does dropping planes bring the packed kernel to at least 0.5x of f32-resident BLAS?

close: 562ce1cbc98f4c1c (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-09_0555)
