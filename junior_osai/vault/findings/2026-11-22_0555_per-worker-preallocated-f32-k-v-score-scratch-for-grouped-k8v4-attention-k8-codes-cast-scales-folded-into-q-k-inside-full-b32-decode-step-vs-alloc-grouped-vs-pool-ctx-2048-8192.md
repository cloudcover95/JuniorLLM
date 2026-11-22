---
local: true
date: 2026-11-22
wave: "2026-11-22 05:55 (ran ~05:43–05:48 MT; no pair reports reached chat this wave, quant derivation lane paused)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "viable (marginal, ctx 8192 step only; attention target missed)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, decode-step, end-to-end, attention, grouped, scratch, preallocation, scale-folding, kv-cache, k8v4, u32-lut, b32, numpy, viable, marginal]
---

# Per-worker f32 K/V/score scratch for grouped K8V4 attention (K8 codes cast, g64 scales folded into q·K) lifts the full B = 32 / ctx 8192 decode step to 1.60–1.62x vs the pool step (was 1.43–1.46x for alloc-grouped this wave), but attention only drops to 62 / 59 ms (target ≤ ~50 ms); scratch alone is just 1.05–1.14x on attention

The 02:55 note ([[2026-11-22_0255_full-b32-decode-step-u32-lut-2-bit-weights-plus-grouped-k8v4-attention-w-clamp-b-8-32-tk1024-vs-int8-vs-hbp8-vs-pool-attention-step-ctx-2048-8192]]) asked: at B = 32 / ctx 8192 the full step is 84 / 101 ms, and ~63–76 ms of that is grouped K8V4 attention (gs.py hbP, W = clamp(B, 8, 32), TK 1024). That kernel allocates fresh f32 K/V tiles (np.empty inside dq8 / dq4) and a fresh score array for every tile in every task. If each task gets a preallocated f32 K/V/score scratch (dequant with out=, exp in place, scales folded into q·K so K8 codes cast straight into the scratch), does B = 32 / ctx 8192 attention drop to ≤ ~50 ms inside the full step, so the u32 + grouped step clears ≥ 1.5x vs the pool step at ctx 8192, at max abs ≤ 1e-5? Answers:
- **Attention ≤ ~50 ms: no.** Inside the full step, B = 32 / ctx 8192 attention is 62.0 / 58.9 ms with scrF (folded scales) and 66.5 / 60.6 ms with scrD (K dequant with out=, no fold), vs 71.0 / 66.4 ms for the 02:55 alloc-grouped kernel and 114 / 108 ms for the pool. Measured attention-only (same q, no weight pass), scratch-vs-alloc is only **1.05–1.14x** (scrF) / 1.08–1.10x (scrD) at B = 32 / ctx 8192, and 1.05–1.20x / 1.01–1.19x across all four points. Allocation was a minor cost. Most of the per-tile time is the arithmetic itself: the K cast, two half-dim batched q·K matmuls over bh = 5 heads, exp, the V4 nibble take + scale, and P·V.
- **Step ≥ 1.5x vs pool at ctx 8192: yes, but only just.** u32 + scrF step 85.3 / 79.5 ms vs pool step 136.2 / 129.0 ms = **1.60–1.62x** (per-round range 1.46–1.78). That's 375 / 402 tok/s vs 235 / 248. scrD gives 1.50–1.58x (per-round 1.43–1.74), and alloc-grouped gives 1.43–1.46x this wave (02:55: 1.48–1.49x). So scrF vs alloc-grouped on the step is 1.11–1.11x. The per-round lows dip below 1.5 for both scratch variants, and the margin over the bar (~0.1x) is about the size of the round-to-round spread. Weight GEMVs + elementwise ops are still a fixed ~21–23 ms.
- **ctx 2048, B = 32 (context):** scrF 1.72–1.75x vs pool step (scrD 1.66–1.68x, alloc-grouped 1.54–1.54x), 865 / 769 tok/s, 92.9 MiB.
- **B scaling at ctx 8192** (scrF vs pool step): 1.27–1.37x at B = 8, 1.47–1.51x at B = 16, 1.60–1.62x at B = 32. Aggregate tok/s 214–225 → 295–309 → 375–402. B = 8 stays below 1.5x and B = 16 straddles it.
- **Error:** scrD is bit-identical to alloc-grouped (max abs vs grp 0.0e+00; vs pool ≤ 5.4e-07). scrF changes rounding because of the folded scales and the two half-dim matmuls: max abs vs pool ≤ **5.3e-06** inside the step (q from the real qkv projection), and min layer cos 0.998832 is unchanged. **Caveat:** a separate attention-only smoke test with gs.py-style q (2·N(0,1)·SC, seed 1, B = 32 / ctx 8192) gave scrF max abs 2.2e-5 vs pool (outputs up to 5.5, about 4e-6 relative), while scrD gave 3.3e-6. So the fold meets the ≤ 1e-5 bar on the step's real q but not for larger-magnitude q. scrD is the only variant that is safe by construction.
- **Bytes:** KV + weights are unchanged (295.40 MiB u32 at ctx 8192, 92.90 at ctx 2048). Scratch adds 46.25 MiB live (8 workers × bh 5 × 1024 × 128 f32 for K and V, plus scores and a u16 nibble buffer). That is per worker, not per task: the W = 32 tasks share the 8 workers' buffers through thread-local storage, which costs 8× the per-task size rather than 32×. int8 weights + scrD clear 1.63–1.71x vs the u32 pool step at 345.1 MiB.

## Method

- Script `step.py` is a copy of the 02:55 `step.py` (`/workspace/bench0255_1122`). It keeps `base.build` + `par.prep / par.fmm` (2-bit u32-LUT, T = 256, 8 workers) and `par.prep8 / par.fmm8` from `/workspace/bench1455_1118`, `step()` verbatim from `dstep.py` (05:55 11/21), and `gs.py` (18:55 11/21, `Stack`, `tiles`, `hbP`) plus `iq.py` (02:55 11/21, KV generator, quantizers, pool `iq.attn`). New this wave: `sk.py` (inline below). It is `gs.hb_range` with per-worker thread-local scratch (`K`, `V`, `S`, `S1` f32, `U` u16), sized bh_max × max(TK, WIN) and sliced per tile. It has two variants: **scrF** casts the K8 codes into the scratch with `np.copyto(casting="unsafe")` and folds the two g64 key scales into q·K (two half-dim batched matmuls with `out=`, scaled in place); **scrD** dequantizes K with `np.multiply(..., out=)` (same math as `iq.dq8`). Both decode V4 with `np.take(LUT16, out=U)` + `np.multiply(out=V)`, handle the V8 window the same way, and do exp in place.
- Variants per point (all grouped ones: W = clamp(B, 8, 32), TK 1024): u32+grp (02:55 kernel), u32+scrF, u32+scrD, u32+pool (one task per (stream, KV head), TK 2048; the baseline), int8+scrD. Grid: ctx 2048 × B 32, and ctx 8192 × B {8, 16, 32}. Same layer shapes, K8V4 format (A8 g64 K, A4 g32 old V, A8 1024-token window) and fixed-n cache as 02:55.
- Error reference is the same as 02:55: exact f32-softmax attention over the unquantized history gives the reference step. Attention max abs is measured vs the pool and vs alloc-grouped on the first batch's q.
- Timing: 5 rounds × (2 warm + 6 timed) full steps per variant, order rotated each round, so 30 timed steps per variant. I report the median, min–max, IQR and per-round-median speedup ranges. Attention-only is also timed: 5 rounds × 3 reps of grp / scrF / scrD / pool on the same q, rotated. OPENBLAS_NUM_THREADS = 1, an 8-thread attention pool plus 8 weight workers, on an 8-core box. Two seeds: run1 11220555, run2 11220255 (= 02:55 run1 seed).
- Command: `cd /workspace/bench0555_1122 && OPENBLAS_NUM_THREADS=1 SEED=11220555 python3 step.py results_run1.json > run1.txt && OPENBLAS_NUM_THREADS=1 SEED=11220255 python3 step.py results_run2.json > run2.txt` (each run takes ~2.5 min).

## Results

### Summary (K8V4, u32-LUT weights, medians, run1; run2)

| ctx | B | step ms scrF / scrD / grp / pool (r1; r2) | attn ms in step scrF / scrD / grp / pool (r1; r2) | scrF vs pool step (r1 / r2, per-round range) | scrD vs pool step (range) | grp vs pool step | scrF vs grp step | scrF tok/s (r1 / r2) | attn share scrF | attn-only ms scrF / scrD / grp / pool (r1; r2) | attn-only scrF/grp, scrD/grp | total MiB u32 (KV) | attn max abs vs pool scrF / scrD | min layer cos scrF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 32 | 37.0 / 39.0 / 42.0 / 64.6; 41.6 / 42.7 / 46.6 / 71.7 | 16.3 / 17.4 / 20.2 / 42.8; 18.6 / 18.9 / 22.1 / 49.6 | **1.75x / 1.72x** (1.60–1.86) | 1.66x / 1.68x (1.53–1.80) | 1.54x / 1.54x | 1.14x / 1.12x | 865 / 769 | 0.45–0.45 | 16.4 / 16.1 / 18.5 / 43.3; 18.1 / 17.5 / 20.2 / 47.8 | 1.11–1.13x, 1.15–1.16x | 92.90 (76.25) | 3.2e-06 / 0.0e+00 | 0.998832 |
| 8192 | 8 | 35.6 / 35.0 / 38.1 / 45.3; 37.4 / 37.5 / 39.4 / 51.1 | 17.9 / 17.3 / 19.4 / 29.1; 19.4 / 19.3 / 21.0 / 32.5 | **1.27x / 1.37x** (0.85–1.48) | 1.29x / 1.36x (1.18–1.45) | 1.19x / 1.30x | 1.07x / 1.05x | 225 / 214 | 0.51–0.51 | 16.3 / 16.5 / 19.6 / 28.5; 20.1 / 21.1 / 23.1 / 32.3 | 1.15–1.20x, 1.10–1.19x | 86.34 (69.69) | 2.2e-06 / 4.2e-07 | 0.999877 |
| 8192 | 16 | 51.7 / 53.8 / 56.3 / 75.8; 54.3 / 65.8 / 63.7 / 81.9 | 32.5 / 34.4 / 36.1 / 56.7; 34.4 / 40.0 / 40.6 / 60.6 | **1.47x / 1.51x** (1.14–1.69) | 1.41x / 1.24x (1.15–1.64) | 1.35x / 1.29x | 1.09x / 1.17x | 309 / 295 | 0.63–0.64 | 32.7 / 37.2 / 37.6 / 57.3; 32.7 / 31.1 / 36.1 / 54.8 | 1.10–1.15x, 1.01–1.16x | 156.02 (139.38) | 5.3e-06 / 5.4e-07 | 0.999415 |
| 8192 | 32 | 85.3 / 90.8 / 94.9 / 136.2; 79.5 / 81.6 / 88.4 / 129.0 | 62.0 / 66.5 / 71.0 / 113.6; 58.9 / 60.6 / 66.4 / 107.9 | **1.60x / 1.62x** (1.46–1.78) | 1.50x / 1.58x (1.43–1.74) | 1.43x / 1.46x | 1.11x / 1.11x | 375 / 402 | 0.73–0.73 | 59.2 / 61.3 / 67.4 / 111.4; 59.5 / 58.0 / 62.4 / 104.6 | 1.05–1.14x, 1.08–1.10x | 295.40 (278.75) | 5.3e-06 / 5.4e-07 | 0.999384 |

### Run 1 (SEED 11220555)

| cfg | step ms med (min–max) | IQR | qkv ms | attn ms | rest ms | attn share | tok/s | vs u32+pool (per-round range) | vs u32+grp (range) | vs int8+scrD (range) | W / TK | total MiB (KV) | layer cos vs f32-attn step | attn max abs vs pool | attn max abs vs grp | attn min cos vs f32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u32+grp_ctx2048_B32 | 41.99 (36.22–52.07) | 5.62 | 3.02 | 20.15 | 19.10 | 0.476 | 762.1 | 1.538x (1.44–1.69) | 1.000x (1.00–1.00) | 0.737x (0.67–0.80) | 32 / 1024 | 92.90 (76.25) | 0.999507 | 0.0e+00 | 0.0e+00 | 0.991451 |
| u32+scrF_ctx2048_B32 | 37.00 (31.80–44.15) | 4.42 | 2.88 | 16.29 | 17.28 | 0.449 | 865.0 | 1.746x (1.60–1.86) | 1.135x (1.06–1.21) | 0.837x (0.76–0.94) | 32 / 1024 | 92.90 (76.25) | 0.999507 | 3.2e-06 | 3.2e-06 | 0.991451 |
| u32+scrD_ctx2048_B32 | 38.97 (33.24–45.97) | 3.86 | 3.14 | 17.37 | 18.46 | 0.449 | 821.1 | 1.657x (1.53–1.78) | 1.077x (0.90–1.16) | 0.794x (0.72–0.90) | 32 / 1024 | 92.90 (76.25) | 0.999507 | 0.0e+00 | 0.0e+00 | 0.991451 |
| u32+pool_ctx2048_B32 | 64.60 (59.78–72.19) | 4.09 | 3.13 | 42.83 | 18.23 | 0.665 | 495.4 | 1.000x (1.00–1.00) | 0.650x (0.59–0.70) | 0.479x (0.46–0.50) | 160 / 2048 | 92.90 (76.25) | 0.999507 | 0.0e+00 | 0.0e+00 | 0.991451 |
| int8+scrD_ctx2048_B32 | 30.96 (27.52–36.85) | 3.58 | 2.44 | 16.03 | 12.41 | 0.520 | 1033.5 | 2.086x (1.98–2.19) | 1.356x (1.24–1.49) | 1.000x (1.00–1.00) | 32 / 1024 | 142.59 (76.25) | 0.997859 | 0.0e+00 | 0.0e+00 | 0.988733 |
| u32+grp_ctx8192_B8 | 38.13 (32.47–51.48) | 6.58 | 2.78 | 19.38 | 15.45 | 0.516 | 209.8 | 1.188x (0.98–1.28) | 1.000x (1.00–1.00) | 0.731x (0.61–0.79) | 8 / 1024 | 86.34 (69.69) | 0.999913 | 4.2e-07 | 0.0e+00 | 0.991945 |
| u32+scrF_ctx8192_B8 | 35.59 (31.17–52.76) | 5.01 | 2.72 | 17.93 | 15.03 | 0.506 | 224.8 | 1.272x (1.11–1.39) | 1.071x (0.99–1.30) | 0.783x (0.71–0.84) | 8 / 1024 | 86.34 (69.69) | 0.999913 | 2.2e-06 | 2.3e-06 | 0.991945 |
| u32+scrD_ctx8192_B8 | 35.01 (29.01–54.07) | 5.64 | 2.68 | 17.30 | 14.64 | 0.504 | 228.5 | 1.293x (1.19–1.39) | 1.089x (1.05–1.22) | 0.796x (0.74–0.89) | 8 / 1024 | 86.34 (69.69) | 0.999913 | 4.2e-07 | 0.0e+00 | 0.991945 |
| u32+pool_ctx8192_B8 | 45.28 (40.29–49.39) | 2.50 | 2.30 | 29.08 | 13.89 | 0.641 | 176.7 | 1.000x (1.00–1.00) | 0.842x (0.78–1.02) | 0.615x (0.59–0.64) | 40 / 2048 | 86.34 (69.69) | 0.999913 | 0.0e+00 | 4.2e-07 | 0.991945 |
| int8+scrD_ctx8192_B8 | 27.86 (24.99–37.07) | 1.55 | 1.73 | 16.43 | 9.38 | 0.589 | 287.1 | 1.625x (1.56–1.69) | 1.369x (1.27–1.64) | 1.000x (1.00–1.00) | 8 / 1024 | 136.02 (69.69) | 0.999318 | 2.0e-06 | 0.0e+00 | 0.989896 |
| u32+grp_ctx8192_B16 | 56.26 (49.93–76.10) | 8.52 | 2.62 | 36.09 | 16.57 | 0.661 | 284.4 | 1.347x (1.08–1.57) | 1.000x (1.00–1.00) | 0.847x (0.66–0.98) | 16 / 1024 | 156.02 (139.38) | 0.999415 | 4.2e-07 | 0.0e+00 | 0.991568 |
| u32+scrF_ctx8192_B16 | 51.71 (45.30–58.33) | 6.39 | 2.86 | 32.46 | 16.16 | 0.630 | 309.4 | 1.465x (1.36–1.59) | 1.088x (0.97–1.39) | 0.921x (0.82–1.00) | 16 / 1024 | 156.02 (139.38) | 0.999415 | 2.2e-06 | 2.3e-06 | 0.991568 |
| u32+scrD_ctx8192_B16 | 53.79 (45.38–63.72) | 6.28 | 2.87 | 34.41 | 16.18 | 0.638 | 297.4 | 1.409x (1.27–1.64) | 1.046x (0.93–1.36) | 0.885x (0.82–1.03) | 16 / 1024 | 156.02 (139.38) | 0.999415 | 4.2e-07 | 0.0e+00 | 0.991568 |
| u32+pool_ctx8192_B16 | 75.78 (69.21–93.97) | 8.66 | 2.72 | 56.66 | 15.95 | 0.749 | 211.1 | 1.000x (1.00–1.00) | 0.742x (0.64–0.92) | 0.628x (0.56–0.65) | 80 / 2048 | 156.02 (139.38) | 0.999415 | 0.0e+00 | 4.2e-07 | 0.991568 |
| int8+scrD_ctx8192_B16 | 47.63 (39.79–61.54) | 5.37 | 2.16 | 33.81 | 11.18 | 0.719 | 335.9 | 1.591x (1.54–1.79) | 1.181x (1.02–1.53) | 1.000x (1.00–1.00) | 16 / 1024 | 205.71 (139.38) | 0.999021 | 2.0e-06 | 0.0e+00 | 0.982076 |
| u32+grp_ctx8192_B32 | 94.91 (83.96–109.59) | 6.07 | 3.53 | 70.99 | 20.50 | 0.748 | 337.2 | 1.435x (1.29–1.54) | 1.000x (1.00–1.00) | 0.879x (0.83–0.92) | 32 / 1024 | 295.40 (278.75) | 0.999415 | 4.8e-07 | 0.0e+00 | 0.991568 |
| u32+scrF_ctx8192_B32 | 85.27 (79.50–97.72) | 6.75 | 3.13 | 62.05 | 19.71 | 0.735 | 375.3 | 1.597x (1.46–1.70) | 1.113x (1.06–1.14) | 0.979x (0.88–1.04) | 32 / 1024 | 295.40 (278.75) | 0.999415 | 2.2e-06 | 2.3e-06 | 0.991568 |
| u32+scrD_ctx8192_B32 | 90.81 (83.06–102.11) | 8.65 | 3.23 | 66.52 | 20.08 | 0.743 | 352.4 | 1.499x (1.43–1.57) | 1.045x (0.99–1.10) | 0.919x (0.82–1.00) | 32 / 1024 | 295.40 (278.75) | 0.999415 | 4.8e-07 | 0.0e+00 | 0.991568 |
| u32+pool_ctx8192_B32 | 136.16 (123.87–159.84) | 11.47 | 3.07 | 113.60 | 19.00 | 0.833 | 235.0 | 1.000x (1.00–1.00) | 0.697x (0.65–0.77) | 0.613x (0.54–0.70) | 160 / 2048 | 295.40 (278.75) | 0.999415 | 0.0e+00 | 4.8e-07 | 0.991568 |
| int8+scrD_ctx8192_B32 | 83.46 (74.91–97.54) | 8.52 | 2.68 | 65.57 | 15.25 | 0.785 | 383.4 | 1.631x (1.43–1.85) | 1.137x (1.08–1.20) | 1.000x (1.00–1.00) | 32 / 1024 | 345.09 (278.75) | 0.999021 | 2.0e-06 | 0.0e+00 | 0.982076 |

| attention only (same q, no weight pass) | grp ms (min–max) | scrF ms | scrD ms | pool ms | scrF/grp | scrD/grp | scrF/pool |
|---|---|---|---|---|---|---|---|
| ctx2048_B32 | 18.55 (16.19–21.84) | 16.43 (14.36–18.25) | 16.12 (13.55–18.74) | 43.30 (39.99–47.66) | 1.128x | 1.151x | 2.635x |
| ctx8192_B8 | 19.61 (16.56–28.28) | 16.32 (14.61–20.39) | 16.46 (14.89–23.20) | 28.51 (26.09–32.67) | 1.202x | 1.191x | 1.747x |
| ctx8192_B16 | 37.64 (30.72–43.01) | 32.69 (28.46–42.11) | 37.23 (28.93–44.92) | 57.26 (54.08–66.00) | 1.151x | 1.011x | 1.752x |
| ctx8192_B32 | 67.36 (61.51–75.28) | 59.18 (55.09–66.22) | 61.26 (57.10–73.59) | 111.38 (105.34–115.89) | 1.138x | 1.100x | 1.882x |

### Run 2 (SEED 11220255)

| cfg | step ms med (min–max) | IQR | qkv ms | attn ms | rest ms | attn share | tok/s | vs u32+pool (per-round range) | vs u32+grp (range) | vs int8+scrD (range) | W / TK | total MiB (KV) | layer cos vs f32-attn step | attn max abs vs pool | attn max abs vs grp | attn min cos vs f32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u32+grp_ctx2048_B32 | 46.62 (40.81–56.52) | 3.01 | 3.44 | 22.12 | 20.70 | 0.481 | 686.4 | 1.537x (1.31–1.65) | 1.000x (1.00–1.00) | 0.752x (0.59–0.81) | 32 / 1024 | 92.90 (76.25) | 0.998832 | 0.0e+00 | 0.0e+00 | 0.991865 |
| u32+scrF_ctx2048_B32 | 41.61 (37.71–49.96) | 3.64 | 3.35 | 18.64 | 19.47 | 0.451 | 769.0 | 1.722x (1.62–1.79) | 1.120x (1.05–1.30) | 0.842x (0.76–0.90) | 32 / 1024 | 92.90 (76.25) | 0.998832 | 2.7e-06 | 2.7e-06 | 0.991865 |
| u32+scrD_ctx2048_B32 | 42.74 (37.03–46.08) | 3.93 | 3.40 | 18.93 | 19.90 | 0.447 | 748.7 | 1.677x (1.66–1.80) | 1.091x (1.03–1.35) | 0.820x (0.77–0.86) | 32 / 1024 | 92.90 (76.25) | 0.998832 | 0.0e+00 | 0.0e+00 | 0.991865 |
| u32+pool_ctx2048_B32 | 71.66 (65.93–84.29) | 4.63 | 3.41 | 49.56 | 19.21 | 0.689 | 446.6 | 1.000x (1.00–1.00) | 0.651x (0.61–0.76) | 0.489x (0.43–0.52) | 160 / 2048 | 92.90 (76.25) | 0.998832 | 0.0e+00 | 0.0e+00 | 0.991865 |
| int8+scrD_ctx2048_B32 | 35.05 (30.24–41.12) | 4.71 | 2.72 | 17.49 | 14.71 | 0.508 | 913.0 | 2.044x (1.94–2.34) | 1.330x (1.24–1.69) | 1.000x (1.00–1.00) | 32 / 1024 | 142.59 (76.25) | 0.998601 | 0.0e+00 | 0.0e+00 | 0.990582 |
| u32+grp_ctx8192_B8 | 39.39 (36.52–45.58) | 3.36 | 2.69 | 20.98 | 15.84 | 0.533 | 203.1 | 1.297x (1.18–1.33) | 1.000x (1.00–1.00) | 0.762x (0.67–0.80) | 8 / 1024 | 86.34 (69.69) | 0.999877 | 4.2e-07 | 0.0e+00 | 0.989156 |
| u32+scrF_ctx8192_B8 | 37.39 (32.29–73.91) | 7.30 | 2.75 | 19.43 | 15.36 | 0.510 | 213.9 | 1.367x (0.85–1.48) | 1.053x (0.64–1.17) | 0.803x (0.50–0.91) | 8 / 1024 | 86.34 (69.69) | 0.999877 | 7.7e-07 | 7.2e-07 | 0.989156 |
| u32+scrD_ctx8192_B8 | 37.54 (33.01–50.55) | 5.60 | 2.77 | 19.32 | 15.59 | 0.514 | 213.1 | 1.361x (1.18–1.45) | 1.049x (0.94–1.16) | 0.800x (0.67–0.88) | 8 / 1024 | 86.34 (69.69) | 0.999877 | 4.2e-07 | 0.0e+00 | 0.989156 |
| u32+pool_ctx8192_B8 | 51.10 (46.59–57.59) | 4.00 | 2.83 | 32.54 | 15.15 | 0.646 | 156.5 | 1.000x (1.00–1.00) | 0.771x (0.75–0.85) | 0.588x (0.57–0.64) | 40 / 2048 | 86.34 (69.69) | 0.999877 | 0.0e+00 | 4.2e-07 | 0.989156 |
| int8+scrD_ctx8192_B8 | 30.04 (25.91–38.93) | 3.76 | 1.86 | 18.05 | 9.73 | 0.610 | 266.3 | 1.701x (1.56–1.77) | 1.311x (1.25–1.49) | 1.000x (1.00–1.00) | 8 / 1024 | 136.02 (69.69) | 0.999268 | 1.3e-06 | 0.0e+00 | 0.987392 |
| u32+grp_ctx8192_B16 | 63.74 (53.84–84.38) | 13.78 | 3.24 | 40.55 | 18.89 | 0.651 | 251.0 | 1.285x (1.10–1.42) | 1.000x (1.00–1.00) | 0.910x (0.79–1.11) | 16 / 1024 | 156.02 (139.38) | 0.999652 | 5.4e-07 | 0.0e+00 | 0.989156 |
| u32+scrF_ctx8192_B16 | 54.28 (45.41–77.84) | 17.84 | 3.01 | 34.43 | 17.33 | 0.639 | 294.8 | 1.509x (1.14–1.69) | 1.174x (1.04–1.24) | 1.068x (0.87–1.38) | 16 / 1024 | 156.02 (139.38) | 0.999652 | 5.3e-06 | 5.3e-06 | 0.989156 |
| u32+scrD_ctx8192_B16 | 65.82 (48.34–77.50) | 19.81 | 3.29 | 40.05 | 19.88 | 0.625 | 243.1 | 1.245x (1.15–1.52) | 0.968x (0.89–1.14) | 0.881x (0.84–0.99) | 16 / 1024 | 156.02 (139.38) | 0.999652 | 5.4e-07 | 0.0e+00 | 0.989156 |
| u32+pool_ctx8192_B16 | 81.92 (69.49–111.37) | 11.92 | 3.10 | 60.55 | 17.68 | 0.749 | 195.3 | 1.000x (1.00–1.00) | 0.778x (0.71–0.91) | 0.708x (0.62–0.82) | 80 / 2048 | 156.02 (139.38) | 0.999652 | 0.0e+00 | 5.4e-07 | 0.989156 |
| int8+scrD_ctx8192_B16 | 57.97 (42.12–98.63) | 16.33 | 2.25 | 41.51 | 13.82 | 0.717 | 276.0 | 1.413x (1.22–1.61) | 1.099x (0.90–1.26) | 1.000x (1.00–1.00) | 16 / 1024 | 205.71 (139.38) | 0.999196 | 1.3e-06 | 0.0e+00 | 0.987392 |
| u32+grp_ctx8192_B32 | 88.39 (80.11–113.84) | 8.38 | 3.33 | 66.37 | 19.15 | 0.753 | 362.0 | 1.459x (1.38–1.61) | 1.000x (1.00–1.00) | 0.853x (0.78–0.91) | 32 / 1024 | 295.40 (278.75) | 0.999384 | 5.4e-07 | 0.0e+00 | 0.989156 |
| u32+scrF_ctx8192_B32 | 79.54 (71.29–86.74) | 4.46 | 3.16 | 58.88 | 18.03 | 0.732 | 402.3 | 1.622x (1.49–1.78) | 1.111x (1.04–1.18) | 0.948x (0.89–0.97) | 32 / 1024 | 295.40 (278.75) | 0.999384 | 5.3e-06 | 5.3e-06 | 0.989156 |
| u32+scrD_ctx8192_B32 | 81.57 (75.28–86.56) | 4.44 | 3.17 | 60.57 | 17.91 | 0.737 | 392.3 | 1.581x (1.53–1.74) | 1.084x (1.04–1.18) | 0.925x (0.86–0.98) | 32 / 1024 | 295.40 (278.75) | 0.999384 | 5.4e-07 | 0.0e+00 | 0.989156 |
| u32+pool_ctx8192_B32 | 128.99 (120.07–152.41) | 6.85 | 3.06 | 107.93 | 18.14 | 0.839 | 248.1 | 1.000x (1.00–1.00) | 0.685x (0.62–0.73) | 0.585x (0.55–0.63) | 160 / 2048 | 295.40 (278.75) | 0.999384 | 0.0e+00 | 5.4e-07 | 0.989156 |
| int8+scrD_ctx8192_B32 | 75.44 (69.00–84.08) | 4.95 | 2.57 | 59.07 | 13.97 | 0.785 | 424.2 | 1.710x (1.59–1.83) | 1.172x (1.10–1.29) | 1.000x (1.00–1.00) | 32 / 1024 | 345.09 (278.75) | 0.998618 | 1.5e-06 | 0.0e+00 | 0.979138 |

| attention only (same q, no weight pass) | grp ms (min–max) | scrF ms | scrD ms | pool ms | scrF/grp | scrD/grp | scrF/pool |
|---|---|---|---|---|---|---|---|
| ctx2048_B32 | 20.20 (18.40–23.78) | 18.14 (17.05–20.04) | 17.46 (16.58–20.68) | 47.82 (45.88–50.32) | 1.114x | 1.157x | 2.637x |
| ctx8192_B8 | 23.12 (20.89–27.11) | 20.13 (18.05–26.40) | 21.09 (17.66–23.90) | 32.28 (31.18–35.71) | 1.148x | 1.096x | 1.604x |
| ctx8192_B16 | 36.06 (31.68–45.99) | 32.68 (28.49–39.31) | 31.06 (28.07–46.18) | 54.83 (52.20–59.83) | 1.104x | 1.161x | 1.678x |
| ctx8192_B32 | 62.44 (57.22–70.82) | 59.51 (53.71–85.05) | 58.05 (53.20–130.48) | 104.61 (100.77–119.98) | 1.049x | 1.076x | 1.758x |

## Verdict

**viable, but marginal, and only for the ctx 8192 step criterion.** With per-worker scratch and folded K scales, the u32 + grouped step is 1.60–1.62x faster than the pool step at B = 32 / ctx 8192 (375 / 402 tok/s, 295.4 MiB + 46.2 MiB scratch). It clears the 1.5x bar that alloc-grouped missed (1.43–1.46x this wave, 1.48–1.49x at 02:55), with max abs ≤ 5.3e-06 on the step's q. However, it misses the stated attention target (≤ ~50 ms; measured 59–62 ms). Scratch by itself is only a 1.05–1.14x attention gain, per-round lows fall under 1.5x, and the folded variant's error exceeds 1e-5 for larger-magnitude q (scrD, which is bit-exact, gives 1.50–1.58x). For i2sd: adopt per-worker scratch (scrD, exact) as a free ~1.05–1.15x on grouped attention. Don't count on it for a robust 1.5x at ctx 8192; attention needs a structural change for that.

## This wave's pair files

Pair builders were idle: no quant pair reports reached chat this wave (the 5:30 quant derivation lane is paused), so there are no pair files to link. I could not verify this against the parent transcript, because no transcript-reading tool was available to this run. Previous note: [[2026-11-22_0255_full-b32-decode-step-u32-lut-2-bit-weights-plus-grouped-k8v4-attention-w-clamp-b-8-32-tk1024-vs-int8-vs-hbp8-vs-pool-attention-step-ctx-2048-8192]].

## Script

sk.py:

    """Scratch grouped K8V4 attention (05:55 11/22): gs.hb_range with zero per-tile allocation of the big arrays.
    Each pool worker owns a preallocated f32 K / V / score scratch (thread-local, sized bh_max x TK x HD), reused by every
    group task that worker runs. K8 codes are cast straight into the K scratch (no scale multiply); the two g64 key scales are
    folded into q.K (two half-dim matmuls into score scratch, scaled in place); V4 nibbles decode via np.take(out=) into a
    u16 scratch and are scaled into the V scratch with out=; V8 window likewise; exp in place."""
    import threading
    import numpy as np
    def make(gs, fold=True):
        iq = gs.iq; KVH, G, HD = iq.KVH, iq.G, iq.HD; LUT16 = iq.LUT16; H2 = HD // 2
        tl = threading.local(); REG = []
        def scr(bh, TK):
            s = getattr(tl, "s", None)
            if s is None or s["bh"] < bh or s["TK"] < TK:
                s = dict(bh=bh, TK=TK, K=np.empty((bh, TK, HD), np.float32), V=np.empty((bh, TK, HD), np.float32),
                         S=np.empty((bh, G, TK), np.float32), S1=np.empty((bh, G, TK), np.float32), U=np.empty((bh, TK, H2), np.uint16))
                tl.s = s; REG.append(s)
            return s
        def hs_range(st, Q, h0, h1, TK):
            bh = h1 - h0; s = scr(bh, max(TK, iq.WIN))
            m = np.full((bh, G, 1), -np.inf, np.float32); l = np.zeros((bh, G, 1), np.float32); acc = np.zeros((bh, G, HD), np.float32)
            Q = np.ascontiguousarray(Q); Q0 = np.ascontiguousarray(Q[:, :, :H2]); Q1 = np.ascontiguousarray(Q[:, :, H2:])
            for a, b, old in gs.tiles(st.n, TK):
                L = b - a; K = s["K"][:bh, :L]; Sx = s["S"][:bh, :, :L]; S1 = s["S1"][:bh, :, :L]; V = s["V"][:bh, :L]
                if fold:
                    np.copyto(K, st.k8[h0:h1, a:b], casting="unsafe")        # K8 codes -> f32, no scale
                    ks = st.ks8[h0:h1, a:b]                                    # (bh, L, 2) g64 scales, folded into q.K
                    np.matmul(Q0, K[:, :, :H2].transpose(0, 2, 1), out=Sx); Sx *= ks[:, None, :, 0]
                    np.matmul(Q1, K[:, :, H2:].transpose(0, 2, 1), out=S1); S1 *= ks[:, None, :, 1]; Sx += S1
                else:                                                          # dequant with out= (same math as iq.dq8)
                    np.multiply(st.k8[h0:h1, a:b].reshape(bh, L, 2, 64), st.ks8[h0:h1, a:b][..., None], out=K.reshape(bh, L, 2, 64))
                    np.matmul(Q, K.transpose(0, 2, 1), out=Sx)
                mn = np.maximum(m, Sx.max(2, keepdims=True)); al = np.exp(m - mn); Sx -= mn; np.exp(Sx, out=Sx)
                l *= al; l += Sx.sum(2, keepdims=True); acc *= al
                if old:
                    U = s["U"][:bh, :L]; np.take(LUT16, st.v4[h0:h1, a:b], out=U)
                    np.multiply(U.view(np.int8).reshape(bh, L, 4, 32), st.v4s[h0:h1, a:b][..., None], out=V.reshape(bh, L, 4, 32))
                else:
                    np.multiply(st.v8[h0:h1, a:b].reshape(bh, L, 2, 64), st.vs8[h0:h1, a:b][..., None], out=V.reshape(bh, L, 2, 64))
                acc += np.matmul(Sx, V); m = mn
            return acc / l
        def hsP(st, Qb, W, TK):
            BH = Qb.shape[0]; cuts = sorted(set(round(i * BH / W) for i in range(W + 1)))
            futs = [iq.POOL.submit(hs_range, st, Qb[h0:h1], h0, h1, TK) for h0, h1 in zip(cuts[:-1], cuts[1:])]
            return np.concatenate([f.result() for f in futs], 0)
        def scratch_bytes(): return sum(sum(v.nbytes for v in s.values() if isinstance(v, np.ndarray)) for s in REG)
        return hsP, scratch_bytes

step.py:

    #!/usr/bin/env python3
    """Scratch grouped K8V4 attention inside the full B = 32 bitnetd decode step, 05:55 11/22. Copy of step.py (02:55 11/22) which reuses:
      - weights: base.build + par.prep/par.fmm (shared u32-LUT 2-bit pass, T=256, W=8) and par.prep8/par.fmm8 (int8 control)
        from /workspace/bench1455_1118, exactly as dstep.py (05:55 11/21, commit 0a0aab5);
      - step(): dstep.py's x -> rms -> fused qkv -> attention -> o -> rms -> gate|up -> silu -> down (qkv / attn / rest timed);
      - attention + lean K8V4 Stack + hbP from gs.py (18:55 11/21), which loads iq.py (02:55 11/21) without main().
    New: sk.py (this wave) = gs.hb_range with per-worker preallocated f32 K/V/score scratch (thread-local, reused by every group task).
    Variants per (ctx, B), all grouped ones W = clamp(B, 8, 32), TK 1024:
                            u32+grp   = 2-bit u32-LUT weights + 02:55 grouped attention (fresh np.empty K/V/score per tile)
                            u32+scrF  = 2-bit weights + scratch grouped, K8 codes cast into scratch, g64 scales folded into q.K
                            u32+scrD  = 2-bit weights + scratch grouped, K dequant with out= (same math as dq8, no fold)
                            u32+pool  = 2-bit weights + one-task-per-(stream, KV head) pool attention, TK 2048 (baseline)
                            int8+scrD = int8 weights + scrD
    Plus attention-only timing (same Q, no weight pass) of grp / scrF / scrD / pool per point.
    Cache held at fixed n (no append). Usage: OPENBLAS_NUM_THREADS=1 SEED=... python3 step.py out.json"""
    import json, sys, time, types
    import numpy as np
    sys.path.insert(0, "/workspace/bench1455_1118")
    import base, par
    sys.path.insert(0, "/workspace/bench0555_1122"); import sk
    from base import rms, silu, D
    gsrc = open("/workspace/bench1855_1121/gs.py").read().rstrip()
    assert gsrc.endswith("main()"); gs = types.ModuleType("gs"); exec(compile(gsrc[:-6], "gs.py", "exec"), gs.__dict__)
    hsF, scrb = sk.make(gs, True); hsD, _ = sk.make(gs, False)
    iq = gs.iq; KVH, G, HD, SC = iq.KVH, iq.G, iq.HD, iq.SC
    TU, W = 256, 8
    GRID = [(2048, 32), (8192, 8), (8192, 16), (8192, 32)]
    ROUNDS, STEPS, WARM = 5, 6, 2
    base.build(); WB = {"u32": par.prep(), "int8": par.prep8()}
    for nm in base.PW:
        for f in ("Wd", "Wt", "wm", "ws", "wm8", "ws8", "PT", "q8"): base.PW[nm].pop(f, None)
    MM = {"u32": par.fmm, "int8": par.fmm8}
    def step(mm, x, af):   # verbatim from dstep.py (05:55 11/21)
        t0 = time.perf_counter(); q = mm("qkv", rms(x), TU, W)[0]
        t1 = time.perf_counter(); att = af(q.reshape(x.shape[0], KVH, G, HD) * SC).reshape(x.shape[0], D)
        t2 = time.perf_counter(); x2 = x + mm("o", att, TU, W)[0]
        g, u = mm("gu", rms(x2), TU, W); y = x2 + mm("down", (silu(g) * u).astype(np.float32), TU, W)[0]
        t3 = time.perf_counter(); return y, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)
    cos = gs.cos
    def main():
        rng = iq.rng; out = {"seed": iq.SEED, "weight_bytes": WB, "cfg": {}, "attn_only": {}}
        print(f"seed {iq.SEED} | weights u32 {WB['u32']/2**20:.2f} MiB int8 {WB['int8']/2**20:.2f} MiB | rounds {ROUNDS} x steps {STEPS} (+{WARM} warm)", flush=True)
        Bmax = max(b for _, b in GRID)
        XS = rng.standard_normal((STEPS, Bmax, D)).astype(np.float32); x0 = XS[0]
        Q0 = {wk: MM[wk]("qkv", rms(x0), TU, W)[0].reshape(Bmax, KVH, G, HD) * SC for wk in MM}
        for n in sorted({g[0] for g in GRID}):
            Bm = max(B for m, B in GRID if m == n); st = gs.Stack(Bm, n)
            REF = {wk: np.empty((Bm * KVH, G, HD), np.float32) for wk in MM}
            for i in range(Bm):
                k, v = iq.kvhist(n); st.add(i, k, v)
                for wk in MM:
                    for h in range(KVH): REF[wk][i * KVH + h] = iq.sm(Q0[wk][i, h], k[h], v[h])
                del k, v
            for B in [B for m, B in GRID if m == n]:
                kvs = st.kvs[:B]; Wg = min(32, max(8, B)); xb = x0[:B]
                AF = {"grp": lambda qq: gs.hbP(st, np.ascontiguousarray(qq.reshape(-1, G, HD)), Wg, 1024),
                      "scrF": lambda qq: hsF(st, np.ascontiguousarray(qq.reshape(-1, G, HD)), Wg, 1024),
                      "scrD": lambda qq: hsD(st, np.ascontiguousarray(qq.reshape(-1, G, HD)), Wg, 1024),
                      "pool": lambda qq: iq.attn(kvs, qq, "k8v4", 2048)}
                RUNS = [("u32", "grp"), ("u32", "scrF"), ("u32", "scrD"), ("u32", "pool"), ("int8", "scrD")]
                kvmib = B * kvs[0].bytes("k8v4") / 2**20; cfgs = {}
                for wk, a in RUNS:
                    ref = REF[wk][:B * KVH]
                    yref = step(MM[wk], xb, lambda qq: ref)[0]
                    ao = np.asarray(AF[a](Q0[wk][:B])).reshape(B * KVH, G, HD)
                    pb = np.asarray(AF["pool"](Q0[wk][:B])).reshape(B * KVH, G, HD)
                    y = step(MM[wk], xb, AF[a])[0]
                    cfgs[(wk, a)] = dict(t=[], lc=min(cos(y[s] - xb[s], yref[s] - xb[s]) for s in range(B)),
                                         amab=float(np.abs(ao - pb).max()), amabg=float(np.abs(ao - np.asarray(AF["grp"](Q0[wk][:B])).reshape(B * KVH, G, HD)).max()), acos=min(cos(ao[i], ref[i]) for i in range(B * KVH)))
                order = list(cfgs)
                for rd in range(ROUNDS):
                    for kk in order[rd % len(order):] + order[:rd % len(order)]:
                        c = cfgs[kk]
                        for i in range(WARM + STEPS):
                            _, t = step(MM[kk[0]], XS[i % STEPS, :B], AF[kk[1]])
                            if i >= WARM: c["t"].append([v * 1e3 for v in t])
                ao_t = {a: [] for a in ("grp", "scrF", "scrD", "pool")}; qa = Q0["u32"][:B]; ao_o = list(ao_t)
                for a in ao_o: AF[a](qa)
                for rd in range(ROUNDS):
                    for a in ao_o[rd % 4:] + ao_o[:rd % 4]:
                        for _ in range(3):
                            t0 = time.perf_counter(); AF[a](qa); ao_t[a].append((time.perf_counter() - t0) * 1e3)
                AO = {a: dict(ms=float(np.median(v)), lo=float(min(v)), hi=float(max(v)), rounds=[float(np.median(v[r * 3:(r + 1) * 3])) for r in range(ROUNDS)]) for a, v in ao_t.items()}
                out["attn_only"][f"ctx{n}_B{B}"] = AO
                print(f"attn-only ctx{n} B{B}: " + " | ".join(f"{a} {d['ms']:.2f} ({d['lo']:.2f}-{d['hi']:.2f})" for a, d in AO.items()) + f" | scrF/grp {AO['grp']['ms']/AO['scrF']['ms']:.3f}x scrD/grp {AO['grp']['ms']/AO['scrD']['ms']:.3f}x scrF/pool {AO['pool']['ms']/AO['scrF']['ms']:.3f}x", flush=True)
                rmed = lambda kk: [float(np.median(np.array(cfgs[kk]["t"])[r * STEPS:(r + 1) * STEPS, 3])) for r in range(ROUNDS)]
                REFS = {"pool": ("u32", "pool"), "grp": ("u32", "grp"), "int8scrD": ("int8", "scrD")}
                ref_rounds = {nm: rmed(kk) for nm, kk in REFS.items()}
                for (wk, a), c in cfgs.items():
                    T = np.array(c["t"]); med = [float(np.median(T[:, j])) for j in range(4)]; rr = rmed((wk, a))
                    key = f"{wk}+{a}_ctx{n}_B{B}"
                    d = dict(B=B, ctx=n, W=B * KVH if a == "pool" else Wg, TK=2048 if a == "pool" else 1024,
                             qkv=med[0], attn=med[1], rest=med[2], step=med[3], lo=float(T[:, 3].min()), hi=float(T[:, 3].max()),
                             iqr=float(np.percentile(T[:, 3], 75) - np.percentile(T[:, 3], 25)), share=float(np.median(T[:, 1] / T[:, 3])),
                             tok_s=B * 1e3 / med[3], kv_mib=kvmib, w_mib=WB[wk] / 2**20, tot_mib=WB[wk] / 2**20 + kvmib,
                             layer_cos=c["lc"], attn_mab_pool=c["amab"], attn_mab_grp=c["amabg"], scratch_mib=scrb() / 2**20, attn_cos_f32=c["acos"], nsteps=len(T))
                    for nm, R in ref_rounds.items():
                        sp = [x / y for x, y in zip(R, rr)]
                        d[f"vs_{nm}"] = float(np.median(np.array(cfgs[REFS[nm]]["t"])[:, 3])) / med[3]
                        d[f"vs_{nm}_lo"], d[f"vs_{nm}_hi"] = min(sp), max(sp)
                    out["cfg"][key] = d
                    print(f"{key:24s} {d['step']:7.2f} ({d['lo']:.2f}-{d['hi']:.2f}) iqr {d['iqr']:5.2f} | qkv {d['qkv']:5.2f} attn {d['attn']:6.2f} rest {d['rest']:5.2f} share {d['share']:.3f} | {d['tok_s']:6.1f} tok/s | vs pool {d['vs_pool']:.3f}x ({d['vs_pool_lo']:.2f}-{d['vs_pool_hi']:.2f}) vs grp {d['vs_grp']:.3f}x ({d['vs_grp_lo']:.2f}-{d['vs_grp_hi']:.2f}) vs int8scrD {d['vs_int8scrD']:.3f}x | tot {d['tot_mib']:.2f} MiB (KV {kvmib:.2f}) | lcos {d['layer_cos']:.6f} amab {d['attn_mab_pool']:.1e} amabg {d['attn_mab_grp']:.1e} acos {d['attn_cos_f32']:.6f}", flush=True)
            del st, REF
        out["scratch_bytes"] = scrb(); json.dump(out, open(sys.argv[1], "w"), indent=1)
    main()

## Next question

Scratch takes allocation out, and B = 32 / ctx 8192 grouped attention is still ~59–62 ms inside the step. Where does the per-tile time go? Time each op of the scrD tile separately (K cast/dequant, q·K, max/exp/sum, V4 take + scale, P·V) at B = 32 / ctx 8192. Then test folding the four g32 V4 scales into P (P·diag(s_g)·codes_g, so the V nibbles are only cast, never scaled) and TK 512 with scratch (L2-resident tiles). Does either bring attention to ≤ ~50 ms and the step to a robust ≥ 1.5x vs pool (per-round low ≥ 1.5) at max abs ≤ 1e-5?

close: d3099a6c4edd3acc
