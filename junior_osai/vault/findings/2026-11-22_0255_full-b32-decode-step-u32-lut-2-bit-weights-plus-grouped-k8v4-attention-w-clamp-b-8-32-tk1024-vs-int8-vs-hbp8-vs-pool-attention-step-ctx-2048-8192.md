---
local: true
date: 2026-11-22
wave: "2026-11-22 02:55 (ran ~02:42–02:55 MT; no pair reports reached chat this wave, quant derivation lane paused)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, decode-step, end-to-end, attention, grouped, head-batched, kv-cache, k8v4, u32-lut, int8, b32, numpy, viable]
---

# Full B = 32 decode step: 2-bit u32-LUT weights + grouped K8V4 attention (W = clamp(B, 8, 32), TK 1024) is 1.54–1.67x faster than the pool-attention step at ctx 2048 but only 1.48–1.49x at ctx 8192 (just under the 1.5x bar). 762 / 870 tok/s at ctx 2048 and 381 / 316 tok/s at ctx 8192, in 92.9 / 295.4 MiB

The 18:55 note asked to re-time the full B = 32 decode step: the shared u32-LUT 2-bit weight pass (05:55 path, [[2026-11-21_0555_full-bitnetd-decode-step-attention-share-u32-lut-2-bit-weights-with-khv4h4-k8v4-a8-kv-vs-int8-a8-b1-b8-ctx-1024-16384]], commit 0a0aab5) plus grouped K8V4 attention (W = clamp(B, 8, 32) groups, TK 1024) at ctx 2048 / 8192. It should be compared with int8 weights + the same attention, the 14:55 hbP8 step (8 groups, TK 2048) and the one-task-per-head pool-attention step. Is the end-to-end step ≥ 1.5x faster than the pool step at B = 32, and what are the aggregate tok/s and total bytes? Answers:
- **ctx 2048, B = 32: yes.** The u32 + grouped step takes 42.0 / 36.8 ms vs 64.7 / 61.6 ms for the pool step, which is **1.54–1.67x** on both seeds. That gives 762 / 870 aggregate tok/s, vs 1.23–1.70x for the hbP8 step.
- **ctx 8192, B = 32: no, it falls just short.** 84.1 / 101.4 ms vs 124.4 / 151.0 ms gives **1.48–1.49x** (per-round range 1.37–1.56), i.e. 381 / 316 tok/s vs 257 / 212. Attention is still 0.75–0.76 of the step (63 / 76 ms, vs 106 / 128 ms for the pool). The weight GEMVs plus elementwise ops add a fixed ~20–25 ms per step, so Amdahl caps the step gain below attention's own ~1.6–1.7x. Inside the full step, both attention kernels run slower than in the 18:55 attention-only bench (pool ~106–128 vs ~97 ms, grouped ~63–76 vs ~58 ms). That is consistent with the weight pass evicting cache between calls; I did not measure it separately.
- **vs the hbP8 step:** 2.49–3.04x at ctx 8192 (hbP8 step 255 / 252 ms, which is 0.49–0.60x of the pool step). Today's fixed hbP8 tiling is an end-to-end regression at B = 32 / ctx 8192, so it should not ship.
- **vs int8 weights + the same attention:** the u32 step is **slower**, at 0.87–0.89x (ctx 2048) and 0.93–0.95x (ctx 8192). The 2-bit LUT expand costs ~4–6 ms more per step than the int8 cast (qkv+rest 20.2–25.0 vs 15.1–19.0 ms at ctx 8192). int8 + grouped does clear 1.5x vs the u32 pool step at ctx 8192 (1.57–1.60x), but it uses more memory.
- **Total bytes (weights + K8V4 KV):** u32 92.90 MiB vs int8 142.59 MiB at ctx 2048 (**1.53x** smaller), and 295.40 vs 345.09 MiB at ctx 8192 (only 1.17x, because the KV (278.75 MiB) dominates). The weights are 16.65 MiB (u32) vs 66.34 MiB (int8) for the full bitnetd layer.
- **B scaling at ctx 8192** (W = clamp(B, 8, 32), so g8 / g16 / g32): grouped vs pool step is 1.23–1.30x at B = 8, 1.32–1.40x at B = 16 and 1.48–1.49x at B = 32. Aggregate tok/s: 219–253 → 296–306 → 316–381.
- **Error:** grouping only changes the float summation order. Attention max abs vs the pool is ≤ 1.4e-06, and the step's layer cos vs the same step with exact f32-softmax attention is unchanged vs the pool step (min 0.998832 for u32 across all points). The u32 weights' layer cos is ≥ int8's at every point.

## Method

- Script `step.py` (inline below) reuses the prior code instead of rewriting it. It takes weights from `base.build` + `par.prep / par.fmm` (shared 2-bit u32-LUT pass, T = 256, 8 workers) and `par.prep8 / par.fmm8` (int8 control, same tiling) in `/workspace/bench1455_1118`, exactly as `dstep.py` (05:55 11/21) does, and its `step()` is copied verbatim from `dstep.py` (x → rms → fused qkv → attention → o → rms → gate|up → silu → down; qkv / attention / rest timed separately). `gs.py` (18:55 11/21) is loaded without `main()` for the lean stacked K8V4 cache, `hbP` (grouped head-batched attention) and, through `iq.py` (02:55 11/21), the KV generator, quantizers and the one-task-per-(stream, KV head) pool (`iq.attn`, TK 2048).
- Layer shapes are the same as those notes: full bitnetd layer (D 2560, F 6912), 8 query heads over 5 KV heads (G 4), HD 128. K8V4 = A8 g64 keys, A4 g32 old values, A8 recent 1024-token window. The cache is held at a fixed n (no append).
- Five steps per point: u32+grp (rule: W = clamp(B, 8, 32), TK 1024), int8+grp, u32+hbP8 (8 groups, TK 2048), u32+pool, int8+pool. Grid: ctx 2048 × B 32, and ctx 8192 × B {8, 16, 32}.
- Error reference: for the first input batch, q is projected once per weight type, and exact f32 softmax attention over the unquantized K/V history gives REF. The reference step replaces attention with REF. Layer cos = min over streams of cos(y − x, y_ref − x).
- Timing: 5 rounds × (2 warmup + 6 timed) steps per variant, with the variant order rotated each round, so 30 timed steps per variant. I report the median, min–max, IQR, and per-round-median speedup ranges. OPENBLAS_NUM_THREADS = 1, an 8-thread attention pool plus 8 weight workers (run one after the other), on an 8-core box. Two seeds: run1 11220255, run2 11211855.
- Command: `cd /workspace/bench0255_1122 && OPENBLAS_NUM_THREADS=1 SEED=11220255 python3 step.py results_run1.json > run1.txt && OPENBLAS_NUM_THREADS=1 SEED=11211855 python3 step.py results_run2.json > run2.txt` (each run takes ~5 min).

## Results

### Summary (K8V4, medians, run1 / run2)

| ctx | B | u32+grp step ms (r1 / r2) | u32+pool step ms | u32+hbP8 step ms | int8+grp step ms | u32+grp vs pool step (r1 / r2, per-round range) | vs hbP8 step | vs int8+grp step | u32+grp tok/s (r1 / r2) | attn share of u32+grp step | total MiB u32 / int8 (KV) | bytes int8 / u32 | min layer cos (u32 / int8) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 32 | 42.02 / 36.80 | 64.69 / 61.56 | 71.56 / 45.18 | 37.21 / 32.11 | **1.54x / 1.67x** (1.35–1.88) | 1.70x / 1.23x | 0.89x / 0.87x | 762 / 870 | 0.46–0.48 | 92.90 / 142.59 (76.25) | 1.53x | 0.998832 / 0.998034 |
| 8192 | 8 | 31.66 / 36.48 | 41.31 / 44.80 | 34.74 / 40.94 | 26.98 / 29.94 | **1.30x / 1.23x** (1.12–1.48) | 1.10x / 1.12x | 0.85x / 0.82x | 253 / 219 | 0.50–0.53 | 86.34 / 136.02 (69.69) | 1.58x | 0.999370 / 0.998547 |
| 8192 | 16 | 52.22 / 54.11 | 69.15 / 75.65 | 78.04 / 69.08 | 44.49 / 49.07 | **1.32x / 1.40x** (1.30–1.60) | 1.49x / 1.28x | 0.85x / 0.91x | 306 / 296 | 0.65–0.66 | 156.02 / 205.71 (139.38) | 1.32x | 0.999370 / 0.998547 |
| 8192 | 32 | 84.09 / 101.39 | 124.37 / 150.96 | 255.23 / 252.15 | 77.85 / 95.91 | **1.48x / 1.49x** (1.37–1.56) | 3.04x / 2.49x | 0.93x / 0.95x | 381 / 316 | 0.75–0.76 | 295.40 / 345.09 (278.75) | 1.17x | 0.999215 / 0.998420 |

### Run 1 (SEED 11220255)

| cfg | step ms med (min–max) | IQR | qkv ms | attn ms | rest ms | attn share | tok/s | vs u32+pool (per-round range) | vs u32+hbP8 (range) | vs int8+grp (range) | W / TK | total MiB (KV) | layer cos vs f32-attn step | attn max abs vs pool | attn min cos vs f32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u32+grp_ctx2048_B32 | 42.02 (33.49–52.44) | 5.35 | 3.52 | 19.93 | 18.79 | 0.464 | 761.5 | 1.539x (1.40–1.72) | 1.703x (1.56–2.02) | 0.886x (0.78–1.49) | 32 / 1024 | 92.90 (76.25) | 0.998832 | 0.0e+00 | 0.991865 |
| int8+grp_ctx2048_B32 | 37.21 (27.54–78.69) | 7.96 | 2.69 | 19.69 | 13.30 | 0.547 | 859.9 | 1.738x (1.08–2.03) | 1.923x (1.36–2.07) | 1.000x (1.00–1.00) | 32 / 1024 | 142.59 (76.25) | 0.998601 | 0.0e+00 | 0.990582 |
| u32+hbP8_ctx2048_B32 | 71.56 (57.53–112.26) | 17.25 | 3.70 | 45.48 | 21.56 | 0.656 | 447.2 | 0.904x (0.79–1.00) | 1.000x (1.00–1.00) | 0.520x (0.48–0.74) | 8 / 2048 | 92.90 (76.25) | 0.998832 | 0.0e+00 | 0.991865 |
| u32+pool_ctx2048_B32 | 64.69 (53.87–82.49) | 10.90 | 3.38 | 43.15 | 17.93 | 0.675 | 494.7 | 1.000x (1.00–1.00) | 1.106x (1.00–1.26) | 0.575x (0.49–0.93) | 160 / 2048 | 92.90 (76.25) | 0.998832 | 0.0e+00 | 0.991865 |
| int8+pool_ctx2048_B32 | 62.14 (52.47–79.23) | 4.43 | 2.75 | 45.08 | 14.48 | 0.724 | 514.9 | 1.041x (0.92–1.14) | 1.152x (0.92–1.32) | 0.599x (0.46–0.95) | 160 / 2048 | 142.59 (76.25) | 0.998601 | 0.0e+00 | 0.990582 |
| u32+grp_ctx8192_B8 | 31.66 (28.15–42.99) | 4.07 | 2.28 | 16.54 | 12.31 | 0.527 | 252.7 | 1.305x (1.14–1.48) | 1.097x (1.01–1.21) | 0.852x (0.75–1.05) | 8 / 1024 | 86.34 (69.69) | 0.999877 | 4.2e-07 | 0.989156 |
| int8+grp_ctx8192_B8 | 26.98 (22.93–47.16) | 3.01 | 1.75 | 16.39 | 8.24 | 0.621 | 296.5 | 1.531x (1.40–1.64) | 1.288x (1.15–1.38) | 1.000x (1.00–1.00) | 8 / 1024 | 136.02 (69.69) | 0.999268 | 1.3e-06 | 0.987392 |
| u32+hbP8_ctx8192_B8 | 34.74 (31.66–42.03) | 2.41 | 2.48 | 18.84 | 13.66 | 0.537 | 230.3 | 1.189x (1.13–1.22) | 1.000x (1.00–1.00) | 0.777x (0.72–0.87) | 8 / 2048 | 86.34 (69.69) | 0.999877 | 0.0e+00 | 0.989156 |
| u32+pool_ctx8192_B8 | 41.31 (38.46–54.45) | 3.08 | 2.17 | 27.07 | 11.79 | 0.660 | 193.7 | 1.000x (1.00–1.00) | 0.841x (0.82–0.89) | 0.653x (0.61–0.71) | 40 / 2048 | 86.34 (69.69) | 0.999877 | 0.0e+00 | 0.989156 |
| int8+pool_ctx8192_B8 | 36.16 (34.38–53.05) | 3.38 | 1.55 | 26.47 | 7.90 | 0.741 | 221.3 | 1.142x (1.08–1.25) | 0.961x (0.90–1.02) | 0.746x (0.69–0.89) | 40 / 2048 | 136.02 (69.69) | 0.999268 | 0.0e+00 | 0.987392 |
| u32+grp_ctx8192_B16 | 52.22 (46.52–105.66) | 5.29 | 2.59 | 34.13 | 14.80 | 0.653 | 306.4 | 1.324x (1.30–1.40) | 1.494x (1.39–1.61) | 0.852x (0.77–0.93) | 16 / 1024 | 156.02 (139.38) | 0.999652 | 5.4e-07 | 0.989156 |
| int8+grp_ctx8192_B16 | 44.49 (38.80–55.75) | 4.92 | 1.87 | 31.97 | 10.14 | 0.739 | 359.6 | 1.554x (1.46–1.69) | 1.754x (1.59–1.86) | 1.000x (1.00–1.00) | 16 / 1024 | 205.71 (139.38) | 0.999196 | 1.3e-06 | 0.987392 |
| u32+hbP8_ctx8192_B16 | 78.04 (72.38–106.14) | 5.21 | 2.66 | 58.50 | 16.54 | 0.754 | 205.0 | 0.886x (0.85–0.93) | 1.000x (1.00–1.00) | 0.570x (0.54–0.63) | 8 / 2048 | 156.02 (139.38) | 0.999652 | 0.0e+00 | 0.989156 |
| u32+pool_ctx8192_B16 | 69.15 (63.12–81.79) | 5.16 | 2.27 | 53.71 | 12.95 | 0.777 | 231.4 | 1.000x (1.00–1.00) | 1.129x (1.07–1.17) | 0.643x (0.59–0.68) | 80 / 2048 | 156.02 (139.38) | 0.999652 | 0.0e+00 | 0.989156 |
| int8+pool_ctx8192_B16 | 66.35 (60.46–130.45) | 5.28 | 1.76 | 53.73 | 9.99 | 0.822 | 241.1 | 1.042x (0.95–1.10) | 1.176x (1.10–1.26) | 0.671x (0.62–0.72) | 80 / 2048 | 205.71 (139.38) | 0.999196 | 0.0e+00 | 0.987392 |
| u32+grp_ctx8192_B32 | 84.09 (77.97–93.09) | 5.98 | 2.93 | 63.27 | 17.28 | 0.760 | 380.5 | 1.479x (1.44–1.54) | 3.035x (2.88–3.52) | 0.926x (0.88–1.01) | 32 / 1024 | 295.40 (278.75) | 0.999384 | 5.4e-07 | 0.989156 |
| int8+grp_ctx8192_B32 | 77.85 (73.64–94.95) | 7.05 | 2.36 | 62.87 | 12.77 | 0.809 | 411.0 | 1.598x (1.47–1.67) | 3.278x (2.88–3.60) | 1.000x (1.00–1.00) | 32 / 1024 | 345.09 (278.75) | 0.998618 | 1.5e-06 | 0.979138 |
| u32+hbP8_ctx8192_B32 | 255.23 (209.71–339.95) | 51.57 | 3.22 | 232.82 | 19.48 | 0.912 | 125.4 | 0.487x (0.41–0.52) | 1.000x (1.00–1.00) | 0.305x (0.28–0.35) | 8 / 2048 | 295.40 (278.75) | 0.999384 | 0.0e+00 | 0.989156 |
| u32+pool_ctx8192_B32 | 124.37 (117.45–142.30) | 5.44 | 2.68 | 105.66 | 15.91 | 0.849 | 257.3 | 1.000x (1.00–1.00) | 2.052x (1.92–2.45) | 0.626x (0.60–0.68) | 160 / 2048 | 295.40 (278.75) | 0.999384 | 0.0e+00 | 0.989156 |
| int8+pool_ctx8192_B32 | 118.49 (110.49–194.48) | 6.21 | 2.09 | 104.80 | 11.24 | 0.888 | 270.1 | 1.050x (0.82–1.08) | 2.154x (1.69–2.61) | 0.657x (0.49–0.72) | 160 / 2048 | 345.09 (278.75) | 0.998618 | 0.0e+00 | 0.979138 |

### Run 2 (SEED 11211855)

| cfg | step ms med (min–max) | IQR | qkv ms | attn ms | rest ms | attn share | tok/s | vs u32+pool (per-round range) | vs u32+hbP8 (range) | vs int8+grp (range) | W / TK | total MiB (KV) | layer cos vs f32-attn step | attn max abs vs pool | attn min cos vs f32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u32+grp_ctx2048_B32 | 36.80 (34.22–59.41) | 3.26 | 2.88 | 17.54 | 16.21 | 0.477 | 869.5 | 1.673x (1.35–1.88) | 1.228x (1.02–1.34) | 0.873x (0.80–0.88) | 32 / 1024 | 92.90 (76.25) | 0.999239 | 0.0e+00 | 0.988585 |
| int8+grp_ctx2048_B32 | 32.11 (28.52–37.63) | 2.99 | 2.20 | 17.27 | 12.17 | 0.539 | 996.6 | 1.917x (1.67–2.18) | 1.407x (1.27–1.57) | 1.000x (1.00–1.00) | 32 / 1024 | 142.59 (76.25) | 0.998034 | 0.0e+00 | 0.989363 |
| u32+hbP8_ctx2048_B32 | 45.18 (39.49–54.71) | 5.17 | 3.21 | 24.12 | 18.28 | 0.518 | 708.3 | 1.363x (1.30–1.46) | 1.000x (1.00–1.00) | 0.711x (0.64–0.79) | 8 / 2048 | 92.90 (76.25) | 0.999239 | 0.0e+00 | 0.988585 |
| u32+pool_ctx2048_B32 | 61.56 (53.81–75.57) | 7.81 | 2.80 | 40.74 | 16.69 | 0.675 | 519.8 | 1.000x (1.00–1.00) | 0.734x (0.68–0.77) | 0.522x (0.46–0.60) | 160 / 2048 | 92.90 (76.25) | 0.999239 | 0.0e+00 | 0.988585 |
| int8+pool_ctx2048_B32 | 54.25 (51.21–67.13) | 3.47 | 2.20 | 40.30 | 11.82 | 0.747 | 589.8 | 1.135x (1.06–1.25) | 0.833x (0.79–0.90) | 0.592x (0.55–0.63) | 160 / 2048 | 142.59 (76.25) | 0.998034 | 0.0e+00 | 0.989363 |
| u32+grp_ctx8192_B8 | 36.48 (30.80–42.64) | 1.58 | 2.50 | 18.32 | 15.49 | 0.503 | 219.3 | 1.228x (1.12–1.29) | 1.122x (1.03–1.18) | 0.821x (0.78–0.97) | 8 / 1024 | 86.34 (69.69) | 0.999370 | 1.4e-06 | 0.990869 |
| int8+grp_ctx8192_B8 | 29.94 (26.18–36.78) | 4.31 | 1.89 | 18.03 | 10.42 | 0.611 | 267.2 | 1.496x (1.28–1.66) | 1.367x (1.17–1.41) | 1.000x (1.00–1.00) | 8 / 1024 | 136.02 (69.69) | 0.998547 | 1.4e-06 | 0.989697 |
| u32+hbP8_ctx8192_B8 | 40.94 (32.79–48.40) | 3.11 | 2.51 | 22.95 | 15.86 | 0.556 | 195.4 | 1.094x (0.95–1.18) | 1.000x (1.00–1.00) | 0.731x (0.71–0.85) | 8 / 2048 | 86.34 (69.69) | 0.999370 | 0.0e+00 | 0.990869 |
| u32+pool_ctx8192_B8 | 44.80 (38.66–53.80) | 4.09 | 2.45 | 27.49 | 14.75 | 0.619 | 178.6 | 1.000x (1.00–1.00) | 0.914x (0.85–1.06) | 0.668x (0.60–0.78) | 40 / 2048 | 86.34 (69.69) | 0.999370 | 0.0e+00 | 0.990869 |
| int8+pool_ctx8192_B8 | 38.74 (33.98–41.42) | 1.87 | 1.63 | 27.32 | 9.40 | 0.715 | 206.5 | 1.157x (1.09–1.26) | 1.057x (1.00–1.16) | 0.773x (0.76–0.90) | 40 / 2048 | 136.02 (69.69) | 0.998547 | 0.0e+00 | 0.989697 |
| u32+grp_ctx8192_B16 | 54.11 (46.68–63.56) | 9.46 | 2.75 | 36.33 | 15.82 | 0.656 | 295.7 | 1.398x (1.34–1.60) | 1.277x (1.09–1.38) | 0.907x (0.76–1.02) | 16 / 1024 | 156.02 (139.38) | 0.999370 | 1.4e-06 | 0.990111 |
| int8+grp_ctx8192_B16 | 49.07 (44.17–67.75) | 5.39 | 2.13 | 35.56 | 11.74 | 0.719 | 326.1 | 1.542x (1.40–1.77) | 1.408x (1.32–1.46) | 1.000x (1.00–1.00) | 16 / 1024 | 205.71 (139.38) | 0.998547 | 1.4e-06 | 0.989030 |
| u32+hbP8_ctx8192_B16 | 69.08 (58.42–89.22) | 5.33 | 2.81 | 48.77 | 18.04 | 0.700 | 231.6 | 1.095x (1.06–1.28) | 1.000x (1.00–1.00) | 0.710x (0.68–0.76) | 8 / 2048 | 156.02 (139.38) | 0.999370 | 0.0e+00 | 0.990111 |
| u32+pool_ctx8192_B16 | 75.65 (69.58–138.28) | 10.97 | 2.77 | 57.03 | 16.42 | 0.750 | 211.5 | 1.000x (1.00–1.00) | 0.913x (0.78–0.95) | 0.649x (0.56–0.71) | 80 / 2048 | 156.02 (139.38) | 0.999370 | 0.0e+00 | 0.990111 |
| int8+pool_ctx8192_B16 | 66.17 (62.86–83.29) | 5.18 | 1.81 | 54.50 | 10.41 | 0.818 | 241.8 | 1.143x (1.10–1.27) | 1.044x (0.95–1.06) | 0.742x (0.68–0.79) | 80 / 2048 | 205.71 (139.38) | 0.998547 | 0.0e+00 | 0.989030 |
| u32+grp_ctx8192_B32 | 101.39 (91.12–182.88) | 6.78 | 3.36 | 76.40 | 21.69 | 0.754 | 315.6 | 1.489x (1.37–1.56) | 2.487x (2.39–2.59) | 0.946x (0.90–1.01) | 32 / 1024 | 295.40 (278.75) | 0.999215 | 1.4e-06 | 0.990111 |
| int8+grp_ctx8192_B32 | 95.91 (88.13–109.84) | 6.09 | 2.75 | 76.24 | 16.27 | 0.799 | 333.7 | 1.574x (1.53–1.69) | 2.629x (2.43–2.70) | 1.000x (1.00–1.00) | 32 / 1024 | 345.09 (278.75) | 0.998420 | 1.9e-06 | 0.989030 |
| u32+hbP8_ctx8192_B32 | 252.15 (207.40–316.85) | 46.34 | 3.41 | 227.59 | 21.11 | 0.901 | 126.9 | 0.599x (0.57–0.63) | 1.000x (1.00–1.00) | 0.380x (0.37–0.41) | 8 / 2048 | 295.40 (278.75) | 0.999215 | 0.0e+00 | 0.990111 |
| u32+pool_ctx8192_B32 | 150.96 (141.06–175.09) | 5.97 | 3.31 | 128.00 | 20.13 | 0.846 | 212.0 | 1.000x (1.00–1.00) | 1.670x (1.57–1.76) | 0.635x (0.59–0.65) | 160 / 2048 | 295.40 (278.75) | 0.999215 | 0.0e+00 | 0.990111 |
| int8+pool_ctx8192_B32 | 142.43 (136.75–153.87) | 5.28 | 2.73 | 125.84 | 14.58 | 0.880 | 224.7 | 1.060x (1.03–1.08) | 1.770x (1.67–1.91) | 0.673x (0.64–0.71) | 160 / 2048 | 345.09 (278.75) | 0.998420 | 0.0e+00 | 0.989030 |

## Verdict

**viable at ctx 2048 only.** At B = 32 / ctx 2048, the end-to-end step with 2-bit u32-LUT weights + grouped K8V4 attention (W = clamp(B, 8, 32), TK 1024) is 1.54–1.67x faster than the pool-attention step (762 / 870 tok/s, 92.9 MiB total, 1.53x smaller than int8), with no extra error. At ctx 8192 it misses the 1.5x bar narrowly (1.48–1.49x). Attention is still ~75% of the step, and the u32 weight pass is 0.93–0.95x of int8's speed, while the memory win there is only 1.17x. For i2sd at B = 32: ship W = clamp(B, 8, 32) / TK 1024 (it replaces hbP8, which is 2.49–3.04x slower end-to-end at ctx 8192). To claim the ctx 8192 step gain, attention needs to get faster.

## This wave's pair files

Pair builders were idle: no quant pair reports reached chat this wave (the 5:30 quant derivation lane is paused), so there are no pair files to link. Previous note: [[2026-11-21_1855_head-group-count-4-32-x-tile-tk-512-2048-sweep-for-grouped-head-batched-k8v4-attention-vs-one-task-per-head-pool-and-hbp8-b8-b16-b32-ctx-2048-8192]].

## Script

step.py:

    #!/usr/bin/env python3
    """End-to-end B = 32 bitnetd decode step with grouped K8V4 attention, 02:55 11/22. Reuses, without rewriting:
      - weights: base.build + par.prep/par.fmm (shared u32-LUT 2-bit pass, T=256, W=8) and par.prep8/par.fmm8 (int8 control)
        from /workspace/bench1455_1118, exactly as dstep.py (05:55 11/21, commit 0a0aab5);
      - step(): dstep.py's x -> rms -> fused qkv -> attention -> o -> rms -> gate|up -> silu -> down (qkv / attn / rest timed);
      - attention + lean K8V4 Stack + hbP from gs.py (18:55 11/21), which loads iq.py (02:55 11/21) without main().
    Variants per (ctx, B):  u32+grp   = 2-bit u32-LUT weights + grouped K8V4 attention, W = clamp(B, 8, 32) groups, TK 1024 (rule)
                            int8+grp  = int8 weights + the same grouped attention
                            u32+hbP8  = 2-bit weights + 14:55 hbP8 (8 groups, TK 2048)
                            u32+pool  = 2-bit weights + one-task-per-(stream, KV head) pool attention, TK 2048 (05:55 step)
                            int8+pool = int8 weights + pool attention
    Cache held at fixed n (no append). Usage: OPENBLAS_NUM_THREADS=1 SEED=... python3 step.py out.json"""
    import json, sys, time, types
    import numpy as np
    sys.path.insert(0, "/workspace/bench1455_1118")
    import base, par
    from base import rms, silu, D
    gsrc = open("/workspace/bench1855_1121/gs.py").read().rstrip()
    assert gsrc.endswith("main()"); gs = types.ModuleType("gs"); exec(compile(gsrc[:-6], "gs.py", "exec"), gs.__dict__)
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
        rng = iq.rng; out = {"seed": iq.SEED, "weight_bytes": WB, "cfg": {}}
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
                      "hbP8": lambda qq: gs.hbP(st, np.ascontiguousarray(qq.reshape(-1, G, HD)), 8, 2048),
                      "pool": lambda qq: iq.attn(kvs, qq, "k8v4", 2048)}
                RUNS = [("u32", "grp"), ("int8", "grp"), ("u32", "hbP8"), ("u32", "pool"), ("int8", "pool")]
                kvmib = B * kvs[0].bytes("k8v4") / 2**20; cfgs = {}
                for wk, a in RUNS:
                    ref = REF[wk][:B * KVH]
                    yref = step(MM[wk], xb, lambda qq: ref)[0]
                    ao = np.asarray(AF[a](Q0[wk][:B])).reshape(B * KVH, G, HD)
                    pb = np.asarray(AF["pool"](Q0[wk][:B])).reshape(B * KVH, G, HD)
                    y = step(MM[wk], xb, AF[a])[0]
                    cfgs[(wk, a)] = dict(t=[], lc=min(cos(y[s] - xb[s], yref[s] - xb[s]) for s in range(B)),
                                         amab=float(np.abs(ao - pb).max()), acos=min(cos(ao[i], ref[i]) for i in range(B * KVH)))
                order = list(cfgs)
                for rd in range(ROUNDS):
                    for kk in order[rd % len(order):] + order[:rd % len(order)]:
                        c = cfgs[kk]
                        for i in range(WARM + STEPS):
                            _, t = step(MM[kk[0]], XS[i % STEPS, :B], AF[kk[1]])
                            if i >= WARM: c["t"].append([v * 1e3 for v in t])
                rmed = lambda kk: [float(np.median(np.array(cfgs[kk]["t"])[r * STEPS:(r + 1) * STEPS, 3])) for r in range(ROUNDS)]
                ref_rounds = {nm: rmed(kk) for nm, kk in (("pool", ("u32", "pool")), ("hbP8", ("u32", "hbP8")), ("int8grp", ("int8", "grp")))}
                for (wk, a), c in cfgs.items():
                    T = np.array(c["t"]); med = [float(np.median(T[:, j])) for j in range(4)]; rr = rmed((wk, a))
                    key = f"{wk}+{a}_ctx{n}_B{B}"
                    d = dict(B=B, ctx=n, W=Wg if a == "grp" else (8 if a == "hbP8" else B * KVH), TK=1024 if a == "grp" else 2048,
                             qkv=med[0], attn=med[1], rest=med[2], step=med[3], lo=float(T[:, 3].min()), hi=float(T[:, 3].max()),
                             iqr=float(np.percentile(T[:, 3], 75) - np.percentile(T[:, 3], 25)), share=float(np.median(T[:, 1] / T[:, 3])),
                             tok_s=B * 1e3 / med[3], kv_mib=kvmib, w_mib=WB[wk] / 2**20, tot_mib=WB[wk] / 2**20 + kvmib,
                             layer_cos=c["lc"], attn_mab_pool=c["amab"], attn_cos_f32=c["acos"], nsteps=len(T))
                    for nm, R in ref_rounds.items():
                        sp = [x / y for x, y in zip(R, rr)]
                        d[f"vs_{nm}"] = float(np.median(np.array(cfgs[{"pool": ("u32", "pool"), "hbP8": ("u32", "hbP8"), "int8grp": ("int8", "grp")}[nm]]["t"])[:, 3])) / med[3]
                        d[f"vs_{nm}_lo"], d[f"vs_{nm}_hi"] = min(sp), max(sp)
                    out["cfg"][key] = d
                    print(f"{key:24s} {d['step']:7.2f} ({d['lo']:.2f}-{d['hi']:.2f}) iqr {d['iqr']:5.2f} | qkv {d['qkv']:5.2f} attn {d['attn']:6.2f} rest {d['rest']:5.2f} share {d['share']:.3f} | {d['tok_s']:6.1f} tok/s | vs pool {d['vs_pool']:.3f}x ({d['vs_pool_lo']:.2f}-{d['vs_pool_hi']:.2f}) vs hbP8 {d['vs_hbP8']:.3f}x ({d['vs_hbP8_lo']:.2f}-{d['vs_hbP8_hi']:.2f}) vs int8grp {d['vs_int8grp']:.3f}x | tot {d['tot_mib']:.2f} MiB (KV {kvmib:.2f}) | lcos {d['layer_cos']:.6f} amab {d['attn_mab_pool']:.1e} acos {d['attn_cos_f32']:.6f}", flush=True)
            del st, REF
        json.dump(out, open(sys.argv[1], "w"), indent=1)
    main()

## Next question

At B = 32 / ctx 8192, the step is 84 / 101 ms, of which ~63–76 ms is grouped attention. The grouped kernel allocates a fresh f32 K and V tile (np.empty inside dq8 / dq4) and a fresh score array for every tile in every task. Give each of the W tasks a preallocated f32 K/V/score scratch (dequant with out=, exp in place, scales folded into q·K so K8 codes cast straight into the scratch). Does that cut B = 32 / ctx 8192 attention to ≤ ~50 ms inside the full step, so the u32 + grouped step clears ≥ 1.5x vs the pool step at ctx 8192, at max abs ≤ 1e-5?

close: 2c43db79a8974f18
