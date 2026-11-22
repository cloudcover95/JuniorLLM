---
local: true
date: 2026-11-21
wave: "2026-11-21 18:55 (ran ~18:47–19:00 MT; no pair reports reached chat this wave, quant derivation lane paused)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, attention, head-batched, group-count, tile-size, kv-cache, k8v4, threads, numpy, viable]
---

# Grouped head-batched K8V4 attention: group count and tile size must scale with B. Today's hbP8 (8 groups, TK 2048) collapses at B = 32 / ctx 8192 (0.42–0.47x of the one-task-per-head pool); 32 groups × TK 1024 is 3.44–3.97x faster than hbP8 and 1.61–1.67x faster than the pool there, near bit-exact

The 14:55 note asked for the best group count (4 / 8 / 16 / 32 pool tasks) and old-token tile size TK (512 / 1024 / 2048) for grouped head-batched K8V4 attention at B = 8 / 16 / 32, and whether the gain over the one-task-per-head pool grows with B (target: ≥ 1.5x over today's hbP8 at B = 32). Answers:
- **Today's hbP8 (g8_tk2048) does not scale.** It is still the best or tied-best at B = 8 (ctx 2048: 2.27–2.36x vs pool; ctx 8192: 1.43–1.46x), but at B = 32 each of its 8 tasks dequantizes a (20 heads, 2048, 128) f32 tile = 20 MiB per K or V block. That drops it to 0.92–1.05x of the pool at ctx 2048 and **0.42–0.47x at ctx 8192** (231 / 217 ms vs pool 97 / 102 ms). g4 is slower than the pool at every point (down to 0.14x).
- **At B = 32, using more groups and smaller tiles fixes it**: g32_tk1024 / g16_tk1024 at ctx 8192 are **3.44–3.99x vs today's hbP8** (target ≥ 1.5x met on both seeds) and 1.61–1.67x vs the 160-task pool (58.3 / 63.2 ms). At ctx 2048 / B = 32, g16_tk2048 / g32_tk1024 reach 2.36–3.15x vs the pool.
- **The gain over the pool does not keep growing with B at ctx 8192**: the best grouped config is ~1.5x at B = 8 (g8_tk1024 1.53–1.62x), ~1.6x at B = 16 (g16_tk1024 1.60–1.73x) and ~1.6–1.7x at B = 32. At ctx 2048 it is larger (~2.2x at B = 8, ~2.7–3.2x at B = 32).
- **Rule that held on every point:** keep (heads per task) × TK at about 5–10K rows, i.e. a per-task f32 K/V tile of ≤ ~5 MiB, and use at least 8 tasks. Concretely: W = clamp(B, 8, 32) groups with TK = 1024 (5 KV heads, i.e. one stream, per task for B = 8–32). Per-point results for this rule are in the summary table.
- Error: grouping only changes float summation order across tile boundaries. Max abs diff vs the pool is 5.2e-06 (0 when TK = 2048), and min cos vs the f32 reference is the same as the pool's (0.968493 worst; the 2x query scale makes attention sharp, as at 14:55). KV bytes are unchanged (K8V4, same codes), so this is a **speed-only** win.

## Method

- Script `gs.py` (inline below) loads `iq.py` from 02:55 11/21 without `main()`, which gives the same KV generator, the A8 g64 K / A4 g32 old-V / A8 1024-token window quantizers, and the `iq.attn` one-task-per-(stream, KV head) pool baseline, TK 2048. It also reuses `hb_range` / `hbP` from `hb.py` (14:55) with W and TK as parameters.
- **The cache is lean and K8V4-only.** Codes and scales are written straight into stacked (B·5, n, …) arrays, and the pool baseline's per-stream KV objects are views into those arrays, so the bytes are the same and nothing is copied. f32 history is dropped after the f32 reference (exact softmax) is taken. This was needed because the 14:55 multi-format KV objects would not fit B = 32 at ctx 8192 in the box's ~2.4 GiB of free RAM.
- Grid: ctx 2048 × B {8, 32} and ctx 8192 × B {8, 16, 32}; 8 query heads over 5 KV heads (G = 4), HD 128. 13 variants per point: pool, plus g{4,8,16,32} × TK {512,1024,2048}. Today's hbP8 is g8_tk2048. The recent 1024-token window is always one tile.
- Timing: an 8-thread pool with OPENBLAS_NUM_THREADS=1. Each point gets 2 warmups, then 7 rounds × 3 reps with the variant order rotated each round. I report the median, min–max, IQR, and per-round-median speedup ranges vs the pool and vs hbP8. There are two independent seeds (run1 11211855, run2 11211455), on an 8-core box.
- Command: `cd /workspace/bench1855_1121 && OPENBLAS_NUM_THREADS=1 SEED=11211855 python3 gs.py results_run1.json > run1.txt && OPENBLAS_NUM_THREADS=1 SEED=11211455 python3 gs.py results_run2.json > run2.txt` (each run takes ~2 min).

## Results

### Summary (K8V4, medians, run1 / run2)

"Rule" is W = clamp(B, 8, 32) groups, TK 1024.

| ctx | B | heads | pool ms (r1 / r2) | today's hbP8 = g8_tk2048 ms (r1 / r2) | hbP8 vs pool | best cfg (r1 / r2) | best ms (r1 / r2) | best vs pool (r1 / r2) | best vs hbP8 (r1 / r2) | rule cfg | rule vs pool (r1 / r2, per-round range) | rule vs hbP8 (r1 / r2, per-round range) | max abs vs pool | min cos vs f32 | KV MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 8 | 40 | 13.56 / 9.30 | 5.96 / 3.95 | 2.27x / 2.36x | g8_tk2048 / g8_tk2048 | 5.96 / 3.95 | **2.27x / 2.36x** | **1.00x / 1.00x** | g8_tk1024 | 2.25x / 2.17x (1.99–2.38) | 0.99x / 0.92x (0.84–1.12) | 2.5e-06 | 0.968493 | 19.06 |
| 2048 | 32 | 160 | 48.04 / 36.80 | 51.99 / 35.05 | 0.92x / 1.05x | g16_tk2048 / g16_tk2048 | 15.24 / 13.79 | **3.15x / 2.67x** | **3.41x / 2.54x** | g32_tk1024 | 2.65x / 2.36x (2.22–2.94) | 2.87x / 2.25x (1.88–3.51) | 5.2e-06 | 0.968493 | 76.25 |
| 8192 | 8 | 40 | 24.52 / 24.85 | 17.11 / 17.02 | 1.43x / 1.46x | g8_tk1024 / g8_tk1024 | 15.98 / 15.30 | **1.53x / 1.62x** | **1.07x / 1.11x** | g8_tk1024 | 1.53x / 1.62x (1.43–1.88) | 1.07x / 1.11x (1.00–1.25) | 2.3e-06 | 0.970413 | 69.69 |
| 8192 | 16 | 80 | 49.76 / 59.79 | 64.97 / 56.10 | 0.77x / 1.07x | g8_tk1024 / g8_tk1024 | 30.14 / 30.30 | **1.65x / 1.97x** | **2.16x / 1.85x** | g16_tk1024 | 1.60x / 1.73x (1.33–2.31) | 2.09x / 1.62x (1.51–3.04) | 4.3e-06 | 0.970413 | 139.38 |
| 8192 | 32 | 160 | 97.19 / 101.52 | 231.42 / 217.37 | 0.42x / 0.47x | g16_tk1024 / g16_tk1024 | 58.05 / 61.33 | **1.67x / 1.66x** | **3.99x / 3.54x** | g32_tk1024 | 1.67x / 1.61x (1.54–1.78) | 3.97x / 3.44x (3.03–4.32) | 4.3e-06 | 0.970413 | 278.75 |

### Speedup vs the one-task-per-head pool, every config (run1 / run2)

| cfg | heads/task @B8 | ctx2048 B8 | ctx2048 B32 | ctx8192 B8 | ctx8192 B16 | ctx8192 B32 |
|---|---|---|---|---|---|---|
| pool | 1 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| g4_tk512 | 10 | 0.75 / 0.78 | 0.69 / 0.58 | 0.88 / 0.91 | 0.73 / 0.92 | 0.67 / 0.69 |
| g4_tk1024 | 10 | 0.71 / 0.71 | 0.65 / 0.55 | 0.84 / 0.82 | 0.60 / 0.77 | 0.42 / 0.42 |
| g4_tk2048 | 10 | 0.77 / 0.76 | 0.64 / 0.53 | 0.52 / 0.53 | 0.37 / 0.39 | 0.14 / 0.16 |
| g8_tk512 | 5 | 1.92 / 1.80 | 1.09 / 1.13 | 1.24 / 1.27 | 1.52 / 1.73 | 1.22 / 1.25 |
| g8_tk1024 | 5 | 2.25 / 2.17 | 0.96 / 1.07 | 1.53 / 1.62 | 1.65 / 1.97 | 0.89 / 0.95 |
| g8_tk2048 | 5 | 2.27 / 2.36 | 0.92 / 1.05 | 1.43 / 1.46 | 0.77 / 1.07 | 0.42 / 0.47 |
| g16_tk512 | 2.5 | 1.43 / 1.40 | 2.53 / 2.21 | 0.86 / 0.89 | 1.31 / 1.35 | 1.53 / 1.47 |
| g16_tk1024 | 2.5 | 1.82 / 1.70 | 2.90 / 2.44 | 1.19 / 1.24 | 1.60 / 1.73 | 1.67 / 1.66 |
| g16_tk2048 | 2.5 | 1.85 / 1.69 | 3.15 / 2.67 | 1.36 / 1.38 | 1.50 / 1.73 | 0.79 / 0.85 |
| g32_tk512 | 1.2 | 0.88 / 0.86 | 2.30 / 2.13 | 0.53 / 0.55 | 0.91 / 0.92 | 1.37 / 1.22 |
| g32_tk1024 | 1.2 | 1.17 / 1.15 | 2.65 / 2.36 | 0.82 / 0.85 | 1.29 / 1.33 | 1.67 / 1.61 |
| g32_tk2048 | 1.2 | 1.14 / 1.16 | 2.73 / 2.39 | 1.07 / 1.08 | 1.48 / 1.56 | 1.62 / 1.59 |

### Run 1 (SEED 11211855)

| cfg | ms med (min–max) | IQR ms | vs pool (per-round range) | vs hbP8 (per-round range) | max abs vs pool | cos vs pool | min cos vs f32 | max abs vs f32 | KV MiB |
|---|---|---|---|---|---|---|---|---|---|
| pool_ctx2048_B8 | 13.56 (10.81–15.06) | 0.67 | 1.000x (1.00–1.00) | 0.440x (0.42–0.52) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 19.06 |
| g4_tk512_ctx2048_B8 | 18.00 (12.40–26.37) | 5.16 | 0.754x (0.62–0.90) | 0.331x (0.28–0.39) | 2.5e-06 | 1.0000000 | 0.968493 | 9.52e-01 | 19.06 |
| g4_tk1024_ctx2048_B8 | 18.98 (14.92–28.71) | 3.88 | 0.715x (0.59–0.86) | 0.314x (0.27–0.38) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 19.06 |
| g4_tk2048_ctx2048_B8 | 17.59 (13.66–87.91) | 7.25 | 0.771x (0.54–0.98) | 0.339x (0.27–0.41) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 19.06 |
| g8_tk512_ctx2048_B8 | 7.05 (5.84–26.38) | 7.73 | 1.924x (1.74–2.17) | 0.846x (0.79–0.97) | 2.5e-06 | 1.0000000 | 0.968493 | 9.52e-01 | 19.06 |
| g8_tk1024_ctx2048_B8 | 6.04 (5.30–18.91) | 1.08 | 2.247x (2.02–2.38) | 0.988x (0.86–1.12) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 19.06 |
| g8_tk2048_ctx2048_B8 | 5.96 (4.72–7.29) | 0.59 | 2.275x (1.91–2.38) | 1.000x (1.00–1.00) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 19.06 |
| g16_tk512_ctx2048_B8 | 9.51 (8.92–11.58) | 0.71 | 1.425x (1.24–1.50) | 0.627x (0.61–0.66) | 2.5e-06 | 1.0000000 | 0.968493 | 9.52e-01 | 19.06 |
| g16_tk1024_ctx2048_B8 | 7.47 (6.61–8.31) | 0.76 | 1.817x (1.75–1.93) | 0.799x (0.75–0.91) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 19.06 |
| g16_tk2048_ctx2048_B8 | 7.32 (6.55–9.90) | 0.60 | 1.851x (1.70–1.98) | 0.814x (0.77–0.90) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 19.06 |
| g32_tk512_ctx2048_B8 | 15.45 (14.84–20.56) | 0.86 | 0.878x (0.62–0.92) | 0.386x (0.33–0.40) | 2.5e-06 | 1.0000000 | 0.968493 | 9.52e-01 | 19.06 |
| g32_tk1024_ctx2048_B8 | 11.57 (11.23–14.31) | 0.75 | 1.173x (0.95–1.22) | 0.516x (0.46–0.54) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 19.06 |
| g32_tk2048_ctx2048_B8 | 11.87 (10.91–16.79) | 0.81 | 1.142x (0.91–1.24) | 0.502x (0.48–0.54) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 19.06 |
| pool_ctx2048_B32 | 48.04 (45.84–55.14) | 3.28 | 1.000x (1.00–1.00) | 1.082x (0.76–1.32) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 76.25 |
| g4_tk512_ctx2048_B32 | 69.30 (54.01–111.40) | 22.29 | 0.693x (0.47–0.80) | 0.750x (0.49–0.80) | 5.2e-06 | 1.0000000 | 0.968493 | 9.52e-01 | 76.25 |
| g4_tk1024_ctx2048_B32 | 74.31 (62.31–126.64) | 39.76 | 0.646x (0.39–0.73) | 0.700x (0.50–0.72) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 76.25 |
| g4_tk2048_ctx2048_B32 | 74.58 (62.64–128.58) | 37.91 | 0.644x (0.44–0.74) | 0.697x (0.47–0.72) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 76.25 |
| g8_tk512_ctx2048_B32 | 43.95 (33.56–49.78) | 7.76 | 1.093x (0.99–1.35) | 1.183x (0.96–1.39) | 5.2e-06 | 1.0000000 | 0.968493 | 9.52e-01 | 76.25 |
| g8_tk1024_ctx2048_B32 | 50.17 (38.11–56.43) | 8.92 | 0.958x (0.87–1.18) | 1.036x (0.81–1.24) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 76.25 |
| g8_tk2048_ctx2048_B32 | 51.99 (38.19–68.41) | 15.05 | 0.924x (0.76–1.32) | 1.000x (1.00–1.00) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 76.25 |
| g16_tk512_ctx2048_B32 | 18.99 (15.76–45.46) | 7.55 | 2.530x (2.45–2.77) | 2.738x (2.10–3.35) | 5.2e-06 | 1.0000000 | 0.968493 | 9.52e-01 | 76.25 |
| g16_tk1024_ctx2048_B32 | 16.59 (14.09–19.79) | 2.66 | 2.895x (2.54–3.49) | 3.133x (2.45–3.69) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 76.25 |
| g16_tk2048_ctx2048_B32 | 15.24 (14.10–17.01) | 1.11 | 3.152x (3.07–3.33) | 3.412x (2.41–4.06) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 76.25 |
| g32_tk512_ctx2048_B32 | 20.88 (18.58–23.04) | 1.34 | 2.301x (2.10–2.57) | 2.490x (1.83–3.00) | 5.2e-06 | 1.0000000 | 0.968493 | 9.52e-01 | 76.25 |
| g32_tk1024_ctx2048_B32 | 18.11 (16.83–20.19) | 0.88 | 2.652x (2.41–2.94) | 2.871x (2.23–3.51) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 76.25 |
| g32_tk2048_ctx2048_B32 | 17.62 (16.34–21.04) | 0.96 | 2.726x (2.65–3.04) | 2.951x (2.12–3.59) | 0.0e+00 | 1.0000000 | 0.968493 | 9.52e-01 | 76.25 |
| pool_ctx8192_B8 | 24.52 (23.52–25.42) | 0.75 | 1.000x (1.00–1.00) | 0.698x (0.66–0.81) | 0.0e+00 | 1.0000000 | 0.980212 | 7.10e-01 | 69.69 |
| g4_tk512_ctx8192_B8 | 27.75 (25.72–33.28) | 2.13 | 0.883x (0.86–0.91) | 0.617x (0.57–0.74) | 1.7e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 69.69 |
| g4_tk1024_ctx8192_B8 | 29.33 (26.23–35.82) | 3.02 | 0.836x (0.75–0.89) | 0.583x (0.52–0.66) | 2.1e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 69.69 |
| g4_tk2048_ctx8192_B8 | 47.56 (42.91–55.89) | 3.94 | 0.516x (0.48–0.53) | 0.360x (0.34–0.42) | 0.0e+00 | 1.0000000 | 0.980212 | 7.10e-01 | 69.69 |
| g8_tk512_ctx8192_B8 | 19.84 (17.65–24.28) | 3.39 | 1.236x (1.10–1.35) | 0.862x (0.77–1.02) | 1.7e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 69.69 |
| g8_tk1024_ctx8192_B8 | 15.98 (14.37–21.38) | 1.58 | 1.534x (1.43–1.73) | 1.071x (1.03–1.25) | 2.1e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 69.69 |
| g8_tk2048_ctx8192_B8 | 17.11 (15.27–23.98) | 3.42 | 1.433x (1.24–1.51) | 1.000x (1.00–1.00) | 0.0e+00 | 1.0000000 | 0.980212 | 7.10e-01 | 69.69 |
| g16_tk512_ctx8192_B8 | 28.40 (26.49–30.72) | 0.96 | 0.863x (0.84–0.91) | 0.602x (0.57–0.68) | 1.7e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 69.69 |
| g16_tk1024_ctx8192_B8 | 20.55 (19.68–24.80) | 1.10 | 1.193x (1.13–1.23) | 0.833x (0.79–0.99) | 2.1e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 69.69 |
| g16_tk2048_ctx8192_B8 | 17.97 (17.26–32.80) | 1.36 | 1.365x (1.25–1.41) | 0.952x (0.88–1.05) | 0.0e+00 | 1.0000000 | 0.980212 | 7.10e-01 | 69.69 |
| g32_tk512_ctx8192_B8 | 46.14 (44.59–48.63) | 1.09 | 0.531x (0.52–0.55) | 0.371x (0.36–0.43) | 1.7e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 69.69 |
| g32_tk1024_ctx8192_B8 | 29.98 (28.53–33.77) | 1.69 | 0.818x (0.78–0.86) | 0.571x (0.55–0.68) | 2.1e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 69.69 |
| g32_tk2048_ctx8192_B8 | 22.83 (22.12–26.68) | 0.93 | 1.074x (1.00–1.12) | 0.749x (0.70–0.86) | 0.0e+00 | 1.0000000 | 0.980212 | 7.10e-01 | 69.69 |
| pool_ctx8192_B16 | 49.76 (47.73–67.96) | 4.00 | 1.000x (1.00–1.00) | 1.306x (1.13–1.57) | 0.0e+00 | 1.0000000 | 0.980212 | 7.10e-01 | 139.38 |
| g4_tk512_ctx8192_B16 | 67.96 (63.70–81.21) | 6.92 | 0.732x (0.67–0.99) | 0.956x (0.91–1.31) | 1.7e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 139.38 |
| g4_tk1024_ctx8192_B16 | 83.50 (78.36–104.87) | 7.14 | 0.596x (0.50–0.75) | 0.778x (0.67–0.99) | 4.3e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 139.38 |
| g4_tk2048_ctx8192_B16 | 135.07 (111.64–248.49) | 26.26 | 0.368x (0.33–0.48) | 0.481x (0.37–0.63) | 0.0e+00 | 1.0000000 | 0.980212 | 7.10e-01 | 139.38 |
| g8_tk512_ctx8192_B16 | 32.81 (28.07–44.15) | 7.48 | 1.517x (1.45–2.22) | 1.980x (1.87–2.93) | 1.7e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 139.38 |
| g8_tk1024_ctx8192_B16 | 30.14 (26.60–53.98) | 4.81 | 1.651x (1.14–2.30) | 2.156x (1.54–3.03) | 4.3e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 139.38 |
| g8_tk2048_ctx8192_B16 | 64.97 (54.72–95.55) | 5.38 | 0.766x (0.63–0.88) | 1.000x (1.00–1.00) | 0.0e+00 | 1.0000000 | 0.980212 | 7.10e-01 | 139.38 |
| g16_tk512_ctx8192_B16 | 37.91 (34.88–69.06) | 3.27 | 1.312x (1.13–1.75) | 1.714x (1.60–2.30) | 1.7e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 139.38 |
| g16_tk1024_ctx8192_B16 | 31.02 (27.48–37.72) | 1.69 | 1.604x (1.33–2.31) | 2.095x (1.89–3.04) | 4.3e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 139.38 |
| g16_tk2048_ctx8192_B16 | 33.21 (28.84–38.35) | 6.04 | 1.498x (1.36–2.33) | 1.957x (1.75–3.06) | 0.0e+00 | 1.0000000 | 0.980212 | 7.10e-01 | 139.38 |
| g32_tk512_ctx8192_B16 | 54.60 (52.57–57.90) | 1.28 | 0.911x (0.87–1.28) | 1.190x (1.07–1.68) | 1.7e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 139.38 |
| g32_tk1024_ctx8192_B16 | 38.57 (36.55–54.77) | 1.53 | 1.290x (0.98–1.77) | 1.684x (1.33–2.33) | 4.3e-06 | 1.0000000 | 0.980212 | 7.10e-01 | 139.38 |
| g32_tk2048_ctx8192_B16 | 33.73 (31.61–45.39) | 1.55 | 1.475x (1.42–1.97) | 1.926x (1.75–2.60) | 0.0e+00 | 1.0000000 | 0.980212 | 7.10e-01 | 139.38 |
| pool_ctx8192_B32 | 97.19 (93.72–103.56) | 2.20 | 1.000x (1.00–1.00) | 2.381x (2.22–2.45) | 0.0e+00 | 1.0000000 | 0.979341 | 7.10e-01 | 278.75 |
| g4_tk512_ctx8192_B32 | 146.13 (137.07–164.58) | 9.52 | 0.665x (0.63–0.70) | 1.584x (1.50–1.72) | 2.4e-06 | 1.0000000 | 0.979341 | 7.10e-01 | 278.75 |
| g4_tk1024_ctx8192_B32 | 231.53 (209.06–297.95) | 11.69 | 0.420x (0.41–0.44) | 0.999x (0.94–1.06) | 4.3e-06 | 1.0000000 | 0.979341 | 7.10e-01 | 278.75 |
| g4_tk2048_ctx8192_B32 | 715.48 (629.91–808.87) | 119.91 | 0.136x (0.12–0.16) | 0.323x (0.28–0.38) | 0.0e+00 | 1.0000000 | 0.979341 | 7.10e-01 | 278.75 |
| g8_tk512_ctx8192_B32 | 79.93 (72.29–111.51) | 6.94 | 1.216x (0.99–1.28) | 2.895x (2.19–3.11) | 2.4e-06 | 1.0000000 | 0.979341 | 7.10e-01 | 278.75 |
| g8_tk1024_ctx8192_B32 | 109.77 (102.87–157.37) | 6.02 | 0.885x (0.83–0.93) | 2.108x (1.94–2.25) | 4.3e-06 | 1.0000000 | 0.979341 | 7.10e-01 | 278.75 |
| g8_tk2048_ctx8192_B32 | 231.42 (200.09–263.02) | 26.40 | 0.420x (0.41–0.45) | 1.000x (1.00–1.00) | 0.0e+00 | 1.0000000 | 0.979341 | 7.10e-01 | 278.75 |
| g16_tk512_ctx8192_B32 | 63.73 (57.95–77.26) | 6.98 | 1.525x (1.45–1.66) | 3.631x (3.42–3.91) | 2.4e-06 | 1.0000000 | 0.979341 | 7.10e-01 | 278.75 |
| g16_tk1024_ctx8192_B32 | 58.05 (54.28–66.43) | 7.01 | 1.674x (1.53–1.77) | 3.986x (3.60–4.35) | 4.3e-06 | 1.0000000 | 0.979341 | 7.10e-01 | 278.75 |
| g16_tk2048_ctx8192_B32 | 123.35 (114.18–134.61) | 6.06 | 0.788x (0.74–0.83) | 1.876x (1.68–2.01) | 0.0e+00 | 1.0000000 | 0.979341 | 7.10e-01 | 278.75 |
| g32_tk512_ctx8192_B32 | 70.83 (67.50–76.85) | 4.80 | 1.372x (1.34–1.45) | 3.267x (3.08–3.52) | 2.4e-06 | 1.0000000 | 0.979341 | 7.10e-01 | 278.75 |
| g32_tk1024_ctx8192_B32 | 58.35 (54.38–64.27) | 3.13 | 1.666x (1.54–1.78) | 3.966x (3.63–4.32) | 4.3e-06 | 1.0000000 | 0.979341 | 7.10e-01 | 278.75 |
| g32_tk2048_ctx8192_B32 | 60.06 (54.07–67.29) | 5.23 | 1.618x (1.56–1.79) | 3.853x (3.53–4.34) | 0.0e+00 | 1.0000000 | 0.979341 | 7.10e-01 | 278.75 |

### Run 2 (SEED 11211455)

| cfg | ms med (min–max) | IQR ms | vs pool (per-round range) | vs hbP8 (per-round range) | max abs vs pool | cos vs pool | min cos vs f32 | max abs vs f32 | KV MiB |
|---|---|---|---|---|---|---|---|---|---|
| pool_ctx2048_B8 | 9.30 (8.81–9.91) | 0.38 | 1.000x (1.00–1.00) | 0.425x (0.41–0.52) | 0.0e+00 | 1.0000000 | 0.989425 | 4.91e-01 | 19.06 |
| g4_tk512_ctx2048_B8 | 11.91 (9.92–13.35) | 1.26 | 0.781x (0.72–0.87) | 0.332x (0.31–0.37) | 2.4e-06 | 1.0000000 | 0.989425 | 4.91e-01 | 19.06 |
| g4_tk1024_ctx2048_B8 | 13.10 (11.69–19.16) | 1.35 | 0.710x (0.66–0.78) | 0.301x (0.27–0.38) | 0.0e+00 | 1.0000000 | 0.989425 | 4.91e-01 | 19.06 |
| g4_tk2048_ctx2048_B8 | 12.27 (10.86–15.45) | 1.47 | 0.758x (0.71–0.81) | 0.322x (0.29–0.37) | 0.0e+00 | 1.0000000 | 0.989425 | 4.91e-01 | 19.06 |
| g8_tk512_ctx2048_B8 | 5.18 (4.43–10.11) | 3.73 | 1.796x (1.68–2.09) | 0.762x (0.74–0.90) | 2.4e-06 | 1.0000000 | 0.989425 | 4.91e-01 | 19.06 |
| g8_tk1024_ctx2048_B8 | 4.28 (3.78–10.12) | 0.63 | 2.173x (1.99–2.38) | 0.922x (0.84–1.08) | 0.0e+00 | 1.0000000 | 0.989425 | 4.91e-01 | 19.06 |
| g8_tk2048_ctx2048_B8 | 3.95 (3.77–5.45) | 0.59 | 2.355x (1.94–2.42) | 1.000x (1.00–1.00) | 0.0e+00 | 1.0000000 | 0.989425 | 4.91e-01 | 19.06 |
| g16_tk512_ctx2048_B8 | 6.64 (6.40–8.78) | 0.73 | 1.401x (1.16–1.45) | 0.595x (0.57–0.60) | 2.4e-06 | 1.0000000 | 0.989425 | 4.91e-01 | 19.06 |
| g16_tk1024_ctx2048_B8 | 5.47 (5.04–7.28) | 0.22 | 1.702x (1.45–1.77) | 0.723x (0.69–0.75) | 0.0e+00 | 1.0000000 | 0.989425 | 4.91e-01 | 19.06 |
| g16_tk2048_ctx2048_B8 | 5.49 (5.20–7.68) | 0.31 | 1.694x (1.49–1.74) | 0.719x (0.68–0.77) | 0.0e+00 | 1.0000000 | 0.989425 | 4.91e-01 | 19.06 |
| g32_tk512_ctx2048_B8 | 10.76 (10.30–11.74) | 0.48 | 0.865x (0.81–0.89) | 0.367x (0.35–0.42) | 2.4e-06 | 1.0000000 | 0.989425 | 4.91e-01 | 19.06 |
| g32_tk1024_ctx2048_B8 | 8.06 (7.76–8.52) | 0.15 | 1.154x (1.07–1.19) | 0.490x (0.47–0.55) | 0.0e+00 | 1.0000000 | 0.989425 | 4.91e-01 | 19.06 |
| g32_tk2048_ctx2048_B8 | 8.05 (7.81–8.37) | 0.24 | 1.155x (1.09–1.19) | 0.490x (0.47–0.60) | 0.0e+00 | 1.0000000 | 0.989425 | 4.91e-01 | 19.06 |
| pool_ctx2048_B32 | 36.80 (35.30–43.47) | 4.71 | 1.000x (1.00–1.00) | 0.953x (0.74–0.99) | 0.0e+00 | 1.0000000 | 0.984462 | 5.09e-01 | 76.25 |
| g4_tk512_ctx2048_B32 | 63.20 (46.42–90.03) | 12.38 | 0.582x (0.46–0.66) | 0.555x (0.44–0.62) | 2.5e-06 | 1.0000000 | 0.984462 | 5.09e-01 | 76.25 |
| g4_tk1024_ctx2048_B32 | 66.95 (54.07–88.53) | 16.50 | 0.550x (0.49–0.72) | 0.524x (0.46–0.60) | 0.0e+00 | 1.0000000 | 0.984462 | 5.09e-01 | 76.25 |
| g4_tk2048_ctx2048_B32 | 69.34 (57.28–103.51) | 11.63 | 0.531x (0.42–0.59) | 0.506x (0.42–0.59) | 0.0e+00 | 1.0000000 | 0.984462 | 5.09e-01 | 76.25 |
| g8_tk512_ctx2048_B32 | 32.55 (28.82–38.61) | 2.56 | 1.130x (1.08–1.34) | 1.077x (0.89–1.10) | 2.5e-06 | 1.0000000 | 0.984462 | 5.09e-01 | 76.25 |
| g8_tk1024_ctx2048_B32 | 34.38 (31.09–38.68) | 2.08 | 1.070x (1.01–1.23) | 1.020x (0.91–1.04) | 0.0e+00 | 1.0000000 | 0.984462 | 5.09e-01 | 76.25 |
| g8_tk2048_ctx2048_B32 | 35.05 (28.78–42.69) | 3.39 | 1.050x (1.01–1.35) | 1.000x (1.00–1.00) | 0.0e+00 | 1.0000000 | 0.984462 | 5.09e-01 | 76.25 |
| g16_tk512_ctx2048_B32 | 16.64 (13.33–24.85) | 8.50 | 2.212x (2.17–2.69) | 2.107x (1.77–2.56) | 2.5e-06 | 1.0000000 | 0.984462 | 5.09e-01 | 76.25 |
| g16_tk1024_ctx2048_B32 | 15.10 (12.50–19.07) | 3.00 | 2.436x (2.32–2.88) | 2.321x (1.72–2.75) | 0.0e+00 | 1.0000000 | 0.984462 | 5.09e-01 | 76.25 |
| g16_tk2048_ctx2048_B32 | 13.79 (12.59–19.99) | 2.77 | 2.669x (2.26–2.99) | 2.542x (1.69–2.81) | 0.0e+00 | 1.0000000 | 0.984462 | 5.09e-01 | 76.25 |
| g32_tk512_ctx2048_B32 | 17.28 (16.21–19.18) | 1.34 | 2.130x (2.13–2.28) | 2.029x (1.65–2.13) | 2.5e-06 | 1.0000000 | 0.984462 | 5.09e-01 | 76.25 |
| g32_tk1024_ctx2048_B32 | 15.61 (13.98–20.86) | 1.73 | 2.358x (2.22–2.69) | 2.246x (1.88–2.42) | 0.0e+00 | 1.0000000 | 0.984462 | 5.09e-01 | 76.25 |
| g32_tk2048_ctx2048_B32 | 15.40 (13.81–17.66) | 1.90 | 2.390x (2.15–2.59) | 2.276x (1.84–2.47) | 0.0e+00 | 1.0000000 | 0.984462 | 5.09e-01 | 76.25 |
| pool_ctx8192_B8 | 24.85 (23.75–26.96) | 1.75 | 1.000x (1.00–1.00) | 0.685x (0.62–0.71) | 0.0e+00 | 1.0000000 | 0.970413 | 5.68e-01 | 69.69 |
| g4_tk512_ctx8192_B8 | 27.20 (25.72–29.47) | 1.81 | 0.914x (0.86–1.02) | 0.626x (0.54–0.66) | 1.5e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 69.69 |
| g4_tk1024_ctx8192_B8 | 30.31 (26.00–33.32) | 3.03 | 0.820x (0.77–0.94) | 0.562x (0.51–0.61) | 2.3e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 69.69 |
| g4_tk2048_ctx8192_B8 | 46.71 (39.29–53.46) | 5.33 | 0.532x (0.51–0.60) | 0.364x (0.34–0.39) | 0.0e+00 | 1.0000000 | 0.970413 | 5.68e-01 | 69.69 |
| g8_tk512_ctx8192_B8 | 19.53 (17.65–24.92) | 4.16 | 1.272x (1.14–1.38) | 0.871x (0.76–0.91) | 1.5e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 69.69 |
| g8_tk1024_ctx8192_B8 | 15.30 (13.87–21.62) | 1.68 | 1.624x (1.54–1.88) | 1.113x (1.00–1.22) | 2.3e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 69.69 |
| g8_tk2048_ctx8192_B8 | 17.02 (14.85–22.57) | 4.17 | 1.460x (1.40–1.60) | 1.000x (1.00–1.00) | 0.0e+00 | 1.0000000 | 0.970413 | 5.68e-01 | 69.69 |
| g16_tk512_ctx8192_B8 | 27.91 (26.89–29.44) | 0.63 | 0.890x (0.88–0.94) | 0.610x (0.56–0.63) | 1.5e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 69.69 |
| g16_tk1024_ctx8192_B8 | 20.10 (19.21–21.90) | 0.52 | 1.236x (1.21–1.34) | 0.847x (0.78–0.89) | 2.3e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 69.69 |
| g16_tk2048_ctx8192_B8 | 18.06 (16.76–18.95) | 0.88 | 1.376x (1.36–1.46) | 0.943x (0.86–1.00) | 0.0e+00 | 1.0000000 | 0.970413 | 5.68e-01 | 69.69 |
| g32_tk512_ctx8192_B8 | 45.00 (43.65–48.29) | 1.08 | 0.552x (0.54–0.60) | 0.378x (0.34–0.40) | 1.5e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 69.69 |
| g32_tk1024_ctx8192_B8 | 29.40 (28.23–35.60) | 0.66 | 0.845x (0.81–0.90) | 0.579x (0.50–0.60) | 2.3e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 69.69 |
| g32_tk2048_ctx8192_B8 | 23.07 (22.00–25.47) | 0.76 | 1.077x (1.05–1.16) | 0.738x (0.66–0.77) | 0.0e+00 | 1.0000000 | 0.970413 | 5.68e-01 | 69.69 |
| pool_ctx8192_B16 | 59.79 (54.16–69.15) | 3.78 | 1.000x (1.00–1.00) | 0.938x (0.85–1.05) | 0.0e+00 | 1.0000000 | 0.970413 | 5.68e-01 | 139.38 |
| g4_tk512_ctx8192_B16 | 65.30 (57.72–76.93) | 2.72 | 0.916x (0.85–1.04) | 0.859x (0.81–0.97) | 1.5e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 139.38 |
| g4_tk1024_ctx8192_B16 | 78.05 (66.62–87.68) | 5.26 | 0.766x (0.68–0.85) | 0.719x (0.65–0.76) | 2.3e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 139.38 |
| g4_tk2048_ctx8192_B16 | 153.35 (130.06–215.23) | 23.73 | 0.390x (0.35–0.42) | 0.366x (0.33–0.39) | 0.0e+00 | 1.0000000 | 0.970413 | 5.68e-01 | 139.38 |
| g8_tk512_ctx8192_B16 | 34.63 (31.11–44.51) | 6.72 | 1.726x (1.58–1.91) | 1.620x (1.50–1.68) | 1.5e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 139.38 |
| g8_tk1024_ctx8192_B16 | 30.30 (27.36–42.26) | 4.31 | 1.973x (1.79–2.30) | 1.851x (1.68–1.94) | 2.3e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 139.38 |
| g8_tk2048_ctx8192_B16 | 56.10 (46.55–62.63) | 3.45 | 1.066x (0.96–1.18) | 1.000x (1.00–1.00) | 0.0e+00 | 1.0000000 | 0.970413 | 5.68e-01 | 139.38 |
| g16_tk512_ctx8192_B16 | 44.26 (42.09–49.72) | 2.58 | 1.351x (1.25–1.49) | 1.268x (1.15–1.37) | 1.5e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 139.38 |
| g16_tk1024_ctx8192_B16 | 34.65 (31.30–38.23) | 2.78 | 1.725x (1.66–1.89) | 1.619x (1.51–1.85) | 2.3e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 139.38 |
| g16_tk2048_ctx8192_B16 | 34.47 (29.90–44.48) | 4.51 | 1.734x (1.70–1.89) | 1.627x (1.58–1.78) | 0.0e+00 | 1.0000000 | 0.970413 | 5.68e-01 | 139.38 |
| g32_tk512_ctx8192_B16 | 64.67 (61.68–69.05) | 2.57 | 0.925x (0.86–1.01) | 0.867x (0.76–0.93) | 1.5e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 139.38 |
| g32_tk1024_ctx8192_B16 | 44.97 (42.88–52.49) | 3.73 | 1.330x (1.14–1.41) | 1.247x (1.11–1.32) | 2.3e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 139.38 |
| g32_tk2048_ctx8192_B16 | 38.30 (35.30–41.96) | 2.93 | 1.561x (1.40–1.67) | 1.465x (1.37–1.56) | 0.0e+00 | 1.0000000 | 0.970413 | 5.68e-01 | 139.38 |
| pool_ctx8192_B32 | 101.52 (97.28–126.86) | 15.98 | 1.000x (1.00–1.00) | 2.141x (1.86–2.49) | 0.0e+00 | 1.0000000 | 0.970413 | 5.68e-01 | 278.75 |
| g4_tk512_ctx8192_B32 | 146.87 (138.75–171.05) | 7.39 | 0.691x (0.64–0.82) | 1.480x (1.22–1.79) | 2.4e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 278.75 |
| g4_tk1024_ctx8192_B32 | 243.83 (215.21–309.85) | 33.49 | 0.416x (0.40–0.47) | 0.891x (0.75–1.08) | 3.3e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 278.75 |
| g4_tk2048_ctx8192_B32 | 642.64 (530.88–762.81) | 64.10 | 0.158x (0.14–0.20) | 0.338x (0.27–0.40) | 0.0e+00 | 1.0000000 | 0.970413 | 5.68e-01 | 278.75 |
| g8_tk512_ctx8192_B32 | 81.43 (72.27–93.71) | 9.43 | 1.247x (1.16–1.49) | 2.670x (2.36–2.92) | 2.4e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 278.75 |
| g8_tk1024_ctx8192_B32 | 106.48 (97.87–122.10) | 7.76 | 0.953x (0.91–1.15) | 2.041x (1.79–2.27) | 3.3e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 278.75 |
| g8_tk2048_ctx8192_B32 | 217.37 (185.48–255.16) | 26.15 | 0.467x (0.40–0.54) | 1.000x (1.00–1.00) | 0.0e+00 | 1.0000000 | 0.970413 | 5.68e-01 | 278.75 |
| g16_tk512_ctx8192_B32 | 69.19 (59.40–92.72) | 15.73 | 1.467x (1.25–1.69) | 3.142x (2.48–3.47) | 2.4e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 278.75 |
| g16_tk1024_ctx8192_B32 | 61.33 (53.91–85.65) | 8.20 | 1.655x (1.52–1.93) | 3.544x (3.04–3.78) | 3.3e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 278.75 |
| g16_tk2048_ctx8192_B32 | 119.22 (109.67–158.78) | 14.88 | 0.852x (0.64–0.97) | 1.823x (1.58–1.90) | 0.0e+00 | 1.0000000 | 0.970413 | 5.68e-01 | 278.75 |
| g32_tk512_ctx8192_B32 | 82.90 (71.80–96.26) | 11.66 | 1.225x (1.19–1.40) | 2.622x (2.39–2.96) | 2.4e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 278.75 |
| g32_tk1024_ctx8192_B32 | 63.17 (56.92–74.44) | 8.90 | 1.607x (1.58–1.77) | 3.441x (3.03–3.94) | 3.3e-06 | 1.0000000 | 0.970413 | 5.68e-01 | 278.75 |
| g32_tk2048_ctx8192_B32 | 63.91 (56.28–78.65) | 8.28 | 1.588x (1.57–1.81) | 3.401x (3.00–4.27) | 0.0e+00 | 1.0000000 | 0.970413 | 5.68e-01 | 278.75 |

## Verdict

**viable** (speed; KV bytes unchanged). At B = 32 / ctx 8192, grouped head-batched K8V4 attention with 16–32 groups and TK 1024 is 3.44–3.99x faster than today's hbP8 and 1.61–1.67x faster than the one-task-per-head pool on both seeds, with max abs diff ≤ 5.2e-06. Today's fixed hbP8 is a regression at B ≥ 16 / ctx 8192 (down to 0.42x of the pool). For i2sd, size the grouping to B: W = clamp(B, 8, 32) tasks, TK 1024, per-task f32 tile ≤ ~5 MiB. Do not ship a fixed 8-group / 2048 tiling. The gain over the pool stays at ~1.5–1.7x at ctx 8192 and does not grow with B.

## This wave's pair files

Pair builders were idle: no quant pair reports reached chat this wave (the 5:30 quant derivation lane is paused), so there are no pair files to link. Previous note: [[2026-11-21_1455_head-batched-attention-batched-matmul-over-all-b-x-5-kv-heads-per-tile-vs-40-task-pool-plus-pool-thread-scaling-1-8-k8v4-khv4h4-a8-kv-b1-b8-ctx-2048-16384]].

## Script

gs.py:

    #!/usr/bin/env python3
    """Group-count x tile-size sweep for grouped head-batched K8V4 attention (hbP), 18:55 11/21. Derived from hb.py (14:55 11/21),
    which loads iq.py (02:55 11/21) without main() for the KV generator, quantizers and the one-task-per-(stream, head) pool baseline.
    Lean K8V4-only cache: codes/scales are built straight into stacked (B*KVH, n, ...) arrays; the per-stream KV objects the pool
    baseline uses are views into them (same bytes, nothing copied), f32 history is dropped after the f32 reference is taken.
    Variants per (ctx, B):  pool   = iq.attn, one pool task per (stream, KV head), TK 2048 (05:55-14:55 baseline)
                            g{W}_tk{T} = hbP with the B*5 heads cut into W contiguous groups (W in 4/8/16/32, one pool task each),
                                         old-token tile TK = T (512/1024/2048); recent 1024-token A8 window is one tile as before.
                            today's hbP8 = g8_tk2048.
    8-thread pool, OPENBLAS_NUM_THREADS=1. Usage: OPENBLAS_NUM_THREADS=1 SEED=... python3 gs.py out.json"""
    import json, sys, time, types
    import numpy as np
    src = open("/workspace/bench0255_1121/iq.py").read().rstrip()
    assert src.endswith("main()"); iq = types.ModuleType("iq"); exec(compile(src[:-6], "iq.py", "exec"), iq.__dict__)
    KVH, G, HD, SC, WIN = iq.KVH, iq.G, iq.HD, iq.SC, iq.WIN
    GRID = [(2048, 8), (2048, 32), (8192, 8), (8192, 16), (8192, 32)]
    GROUPS = (4, 8, 16, 32); TKS = (512, 1024, 2048); BASE_TK = 2048; ROUNDS, REPS = 7, 3
    class Stack:
        def __init__(s, Bm, n):
            BH = Bm * KVH; s.n = n
            s.k8 = np.empty((BH, n, HD), np.int8); s.ks8 = np.empty((BH, n, 2), np.float32)
            s.v8 = np.empty((BH, n, HD), np.int8); s.vs8 = np.empty((BH, n, 2), np.float32)
            s.v4 = np.empty((BH, n, HD // 2), np.uint8); s.v4s = np.empty((BH, n, 4), np.float32); s.kvs = []
        def add(s, i, k, v):
            sl = slice(i * KVH, (i + 1) * KVH)
            (s.k8[sl], s.ks8[sl]), (s.v8[sl], s.vs8[sl]) = iq.q8(k), iq.q8(v)
            s.v4[sl], s.v4s[sl] = iq.q4(iq.dq8(s.v8[sl], s.vs8[sl]))
            kv = iq.KV.__new__(iq.KV); kv.n = s.n
            for a in ("k8", "ks8", "v8", "vs8", "v4", "v4s"): setattr(kv, a, getattr(s, a)[sl])
            s.kvs.append(kv)
    def tiles(n, TK):
        wb = max(0, n - WIN)
        return [(a, min(a + TK, wb), True) for a in range(0, wb, TK)] + [(a, min(a + WIN, n), False) for a in range(wb, n, WIN)]
    def hb_range(st, Q, h0, h1, TK):
        bh = h1 - h0; m = np.full((bh, G, 1), -np.inf, np.float32); l = np.zeros((bh, G, 1), np.float32); acc = np.zeros((bh, G, HD), np.float32)
        for a, b, old in tiles(st.n, TK):
            K = iq.dq8(st.k8[h0:h1, a:b], st.ks8[h0:h1, a:b]); Sx = np.matmul(Q, K.transpose(0, 2, 1))
            mn = np.maximum(m, Sx.max(2, keepdims=True)); al = np.exp(m - mn); Sx -= mn; np.exp(Sx, out=Sx)
            l *= al; l += Sx.sum(2, keepdims=True); acc *= al
            V = iq.dq4(st.v4[h0:h1, a:b], st.v4s[h0:h1, a:b]) if old else iq.dq8(st.v8[h0:h1, a:b], st.vs8[h0:h1, a:b])
            acc += np.matmul(Sx, V); m = mn
        return acc / l
    def hbP(st, Qb, W, TK):
        BH = Qb.shape[0]; cuts = sorted(set(round(i * BH / W) for i in range(W + 1)))
        futs = [iq.POOL.submit(hb_range, st, Qb[h0:h1], h0, h1, TK) for h0, h1 in zip(cuts[:-1], cuts[1:])]
        return np.concatenate([f.result() for f in futs], 0)
    def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    def main():
        rng = iq.rng; out = {"seed": iq.SEED, "cfg": {}}
        print(f"seed {iq.SEED} | groups {GROUPS} | TK {TKS} | rounds {ROUNDS} x reps {REPS} | pool {iq.W} threads", flush=True)
        for n in sorted({g[0] for g in GRID}):
            Bm = max(B for m, B in GRID if m == n); st = Stack(Bm, n)
            Qall = (rng.standard_normal((Bm, KVH, G, HD)).astype(np.float32) * 2.0) * SC
            REF = np.empty((Bm * KVH, G, HD), np.float32)
            for i in range(Bm):
                k, v = iq.kvhist(n); st.add(i, k, v)
                for h in range(KVH): REF[i * KVH + h] = iq.sm(Qall[i, h], k[h], v[h])
                del k, v
            for B in [B for m, B in GRID if m == n]:
                kvs = st.kvs[:B]; q = Qall[:B]; Qb = np.ascontiguousarray(q.reshape(B * KVH, G, HD)); ref = REF[:B * KVH]
                base = iq.attn(kvs, q, "k8v4", BASE_TK).reshape(B * KVH, G, HD)
                variants = [("pool", lambda: iq.attn(kvs, q, "k8v4", BASE_TK))]
                variants += [(f"g{W}_tk{T}", (lambda W=W, T=T: hbP(st, Qb, W, T))) for W in GROUPS for T in TKS]
                cfgs = {}
                for nm, f in variants:
                    o = np.asarray(f()).reshape(B * KVH, G, HD)
                    cfgs[nm] = dict(f=f, t=[], mab=float(np.abs(o - base).max()), cb=min(cos(o[i], base[i]) for i in range(B * KVH)),
                                    cr=min(cos(o[i], ref[i]) for i in range(B * KVH)), mar=float(np.abs(o - ref).max()))
                order = list(cfgs)
                for _ in range(2):
                    for k in order: cfgs[k]["f"]()
                for rd in range(ROUNDS):
                    for k in order[rd % len(order):] + order[:rd % len(order)]:
                        for _ in range(REPS):
                            t0 = time.perf_counter(); cfgs[k]["f"](); cfgs[k]["t"].append((time.perf_counter() - t0) * 1e3)
                kvmib = B * kvs[0].bytes("k8v4") / 2**20
                rb = lambda nm: [float(np.median(cfgs[nm]["t"][i * REPS:(i + 1) * REPS])) for i in range(ROUNDS)]
                P, H8 = rb("pool"), rb("g8_tk2048")
                for nm, c in cfgs.items():
                    T = np.array(c["t"]); med = float(np.median(T)); r = rb(nm)
                    sp_p = [x / y for x, y in zip(P, r)]; sp_h = [x / y for x, y in zip(H8, r)]
                    d = dict(var=nm, ctx=n, B=B, ms=med, lo=float(T.min()), hi=float(T.max()), iqr=float(np.percentile(T, 75) - np.percentile(T, 25)),
                             vs_pool=float(np.median(cfgs["pool"]["t"])) / med, vs_pool_lo=min(sp_p), vs_pool_hi=max(sp_p),
                             vs_hbP8=float(np.median(cfgs["g8_tk2048"]["t"])) / med, vs_hbP8_lo=min(sp_h), vs_hbP8_hi=max(sp_h),
                             mab_base=c["mab"], cos_base=c["cb"], cos_f32=c["cr"], mab_f32=c["mar"], kv_mib=kvmib, n=len(T))
                    out["cfg"][f"{nm}_ctx{n}_B{B}"] = d
                    print(f"{nm+f'_ctx{n}_B{B}':26s} {med:8.2f} ({d['lo']:.2f}-{d['hi']:.2f}) iqr {d['iqr']:5.2f} | vs pool {d['vs_pool']:.3f}x ({min(sp_p):.2f}-{max(sp_p):.2f}) | vs hbP8 {d['vs_hbP8']:.3f}x ({min(sp_h):.2f}-{max(sp_h):.2f}) | mab {c['mab']:.1e} cosf32 {c['cr']:.6f} | KV {kvmib:.2f} MiB", flush=True)
            del st, REF
        json.dump(out, open(sys.argv[1], "w"), indent=1)
    main()

## Next question

The B-sized grouped kernel makes attention ~1.6x faster at B = 32 / ctx 8192 (~60 ms vs ~97 ms). Re-time the full B = 32 decode step: the shared u32-LUT 2-bit weight pass (05:55 path) plus grouped K8V4 attention (W = clamp(B, 8, 32), TK 1024), at ctx 2048 / 8192. Compare it with int8 weights + the same attention, and with today's hbP8 step. Is the end-to-end step ≥ 1.5x faster than the pool-attention step at B = 32, and what is the aggregate tok/s and total bytes (weights + KV)?

close: 23be3211ff2b79e8
