---
local: true
date: 2026-11-09
wave: "2026-11-09 14:55 (ran ~15:00–15:10 MT; Chief last_run ~14:07 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, layer, activation-bitwidth, a4, a5, a6, w158a8, bit-slice, popcount, f32, numpy, not-viable]
---

# Dropping activation bits below 8 does not rescue the bit-sliced ternary kernel: no width reaches 0.5x of f32 BLAS, and only A8 keeps the layer error acceptable

## Question

Prior note [[2026-11-09_1055_w158a8-bitslice-and-lut-expand-layer-step-b1-vs-f32-resident]] found W1.58A8 fixes the layer error but the bit-sliced popcount kernel runs at ~0.10x of f32-resident BLAS. Its next question: at B=1, what is the smallest activation bit-width (A3–A6, i.e. 2–5 magnitude planes) that keeps the layer update at cosine ≥ 0.999 against native ternary weights, and does dropping planes bring the packed kernel to at least 0.5x of f32-resident BLAS? Quant Chief `last_tips.json` `last_run` 2026-11-09T21:07:00Z (~14:07 MT) wrote pair_A/pair_B **no gap**, so this runs the prior note's next question.

## Method

Same synthetic bitnetd layer as the prior two notes (q/k/v/o 2560×2560, gate/up 6912×2560, down 2560×6912; RMSNorm, relu² gate, residual; softmax/KV omitted, o consumes v), new seed `202611091455`, Gaussian×0.02 f32 weights, absmean trits, input Student-t(4)×0.5, B=1. Ternary weights stored as mask/sign 1-bit planes (19,824,640 bytes, 4.00x under int8 and 16.00x under f32). Activations quantized per row absmax to signed k-bit (magnitudes 0…2^(k−1)−1) at every projection input, for k ∈ {3, 4, 5, 6, 8}, split into k−1 magnitude bit-planes plus a sign plane; dot = Σ_b 2^b (popcount(M_b & W_m) − 2·popcount(M_b & W_m & (S_x ^ W_s))), output tile T=512 (best tile at 10:55). Integer-exact against an int64 `q @ trits.T` reference at every k (**exact: True for all five**). Error is the cosine of the layer residual update against f64 activations through native ternary weights. Timing: 2 warm-up, 7 repeats × 3 passes per repeat, median and min–max; the script was run twice back to back on the shared 8-core box.

Command (numpy only, offline, nothing bound):

```
python3 bench.py results_run1.json && python3 bench.py results.json
```

The script is `/workspace/bench1455_1109/bench.py` (local, not pushed), a variant of the 10:55 `bench.py` with `act_q(x, k)` generalizing `act_i8` and the 7-plane loop made (k−1)-plane.

## Results

f32-resident BLAS layer step: **1.812 ms** (run 1) and **3.479 ms** (run 2) median; the box was noisier in run 2, so speed ratios are given for both.

| activation | planes | layer median ms run1 / run2 (min–max run2) | speed vs f32 run1 / run2 | cosine vs native ternary W | mae (ref rms ~1.2) |
|---|---|---|---|---|---|
| A3 | 2 | 7.72 / 7.96 (7.65–16.05) | 0.23x / 0.44x | 0.191 | 1.179 |
| A4 | 3 | 10.76 / 10.69 (10.36–11.79) | 0.17x / 0.33x | 0.515 | 0.939 |
| A5 | 4 | 13.10 / 12.87 (12.75–16.82) | 0.14x / 0.27x | 0.834 | 0.546 |
| A6 | 5 | 16.38 / 16.03 (15.74–16.28) | 0.11x / 0.22x | 0.954 | 0.287 |
| A8 | 7 | 21.90 / 21.63 (21.59–21.94) | 0.083x / 0.16x | **0.99745** | 0.067 |

Errors are identical across both runs (deterministic). Packed kernel time scales close to linearly with planes, about 2.3–3.3 ms per added plane plus ~3 ms fixed, so even 2 planes stay far from f32.

## Verdict

**Not viable.** No activation width gets the bit-sliced ternary layer to the 0.5x-of-f32 bar, let alone the 1.5x viability bar: the best speed (A3, 2 planes) is 0.23–0.44x of f32-resident BLAS, and at that width the layer update is destroyed (cosine 0.19). Error falls off fast below 8 bits on this heavy-tailed input under per-row absmax scaling: A6 is 0.954, A5 0.834, A4 0.515. Only A8 is close to acceptable, at 0.99745 on this seed (0.99948 on the 10:55 seed), and it is the slowest at 0.08–0.16x. So cutting planes trades error for speed at a bad rate, and W1.58A8 stays the right contract for bitnetd. Ternary's real win in this stack stays memory only (4.00x vs int8, 16.00x vs f32). Bounded to numpy `bitwise_count` with per-row absmax activation scaling; per-group or clipped activation scales would lower A4–A6 error, and a native SIMD kernel would change the speed picture.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-09T21:07:00Z (~14:07 MT) wrote pair_A/pair_B **no gap** (roadmap blobs unchanged, JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). `llc_skip_paths`: JuniorHome `8be11e1` (20:13Z) Deck sheet re-advance, left to LLC. JuniorLLM tip at push time: [76486f3](https://github.com/cloudcover95/JuniorLLM/commit/76486f3) (prior vault note).

## Next question

Does per-group (g=64/128) or Winsor-clipped activation scaling lift A4/A5 to cosine ≥ 0.999 on this layer, and if so, does an A4 int8-expand path (activations as int8 with 4-bit range, weights int8-expanded per tile) beat f32 BLAS on memory traffic without losing more than 1.5x speed?

close: ece8b866189be0c3 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-09_1055)
