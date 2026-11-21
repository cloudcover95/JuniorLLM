---
local: true
date: 2026-11-21
wave: "2026-11-21 05:55 (ran ~05:50–06:05 MT; no pair reports reached chat this wave, quant derivation lane paused)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-stream, full-layer, attention-share, kv-cache, khv4h4, k8v4, a8-g64, u32-lut, int8-weights, numpy, not-viable]
---

# In the full decode step, attention is not small: at B = 8 it is 30–68% of each step, so khv4h4's slower attention costs 3–19% of tok/s for 1–3% fewer bytes

The 02:55 note asked whether khv4h4's ~0.85x attention speed vs K8V4 would vanish end to end (under 4% tok/s if attention were under 25% of the step). It does not at B = 8:
- **Weight GEMVs are flat**: 2-bit u32-LUT qkv + o + gate/up/down take **~15.5–18.5 ms per step at B = 8 and ~21.5–24 ms at B = 1**, independent of ctx. int8-resident weights take ~10–12 ms (B = 8) and ~15 ms (B = 1).
- **Attention share at B = 8** with khv4h4: **0.30–0.33 at ctx 1024, 0.44–0.46 at 2048, 0.54–0.56 at 4096, 0.68 at 8192**. It crosses 50% at **ctx ≈ 3,000** (linear fit; K8V4 ≈ 3,850–4,050, all-A8 ≈ 5,050–5,550).
- **At B = 1** it stays small: 0.06 at 1024, 0.20 at 8192, 0.31–0.32 at 16384. It crosses 25% at ctx ≈ 10,400–12,700 and 50% only at ≈ 35–44k.
- **End-to-end cost of khv4h4 vs K8V4** at B = 8: step **0.90–0.91x at ctx 2048, 0.87–0.97x at 4096, 0.81–0.90x at 8192**, for **1.3–2.8% fewer total bytes**. At B = 1 it is 0.96–1.00x (free, but there is also no real byte saving there).
- **Ternary vs int8 end to end**: u32 + khv4h4 holds the **1.80x total-memory win vs int8 + A8 at B = 8 / ctx 8192 (83.99 vs 151.34 MiB)**, 2.1–3.8x at shorter ctx, but runs at **0.64–0.66x of int8 + A8 tok/s** at ctx 8192 and 0.58–0.73x elsewhere. That is the same memory-for-speed trade as 11/18, now with the attention cost made explicit.

Verdict: **not-viable** for the question asked. The premise (attention under ~25% of the step) holds only at B = 1 below ctx ≈ 10k. At B = 8, which is where the cache bytes matter, attention dominates past ctx ≈ 3k, so khv4h4's 2.8% byte saving over K8V4 costs up to ~19% tok/s. For i2sd: K8V4 stays the default cache. khv4h4 is only worth it at B = 1, where it saves almost nothing. The 1.80x memory win is real and unchanged but is not new.

## Question

[[2026-11-21_0255_int8-code-k-scores-p-side-v-scales-and-4-hot-k-channels-for-khv4h-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-and-all-a8-u32-lut-total-bytes]] found khv4h4 (4 hot K channels f16 + A8 g64 K with f16 scales + A4 g32 old V, A8 1024 window) is the smallest accurate cache (83.99 MiB at B = 8 / ctx 8192, 0.973x K8V4, 1.802x below int8 + A8), but its attention runs at 0.82–0.94x of K8V4. Its next question, on the full bitnetd layer with u32-LUT 2-bit weights:
- What share of each decode step's wall time goes to attention vs the 2-bit weight GEMVs (QKV, O, gate/up/down)?
- If attention is under ~25% of the step, the 0.85x attention speed costs < 4% tok/s. Is the 1.80x memory win then the deciding factor for i2sd?
- At what ctx does attention cross 50% of the step?

## Method

- `dstep.py` (below) composes the 11/18 14:55 full-layer weight path ('/workspace/bench1455_1118' base.py + par.py: 2-bit pack, u32-LUT expand, T = 256, W = 8 threads, fused qkv / gate|up sites; int8-resident control 'par.fmm8') with the 02:55 11/21 attention ('/workspace/bench0255_1121/iq.py', loaded without its 'main()': online softmax over TK tiles, 8-thread pool over (stream, KV head), A8 1024 window).
- Step: x → rms → qkv → attention over each stream's cache → o → rms → gate|up → silu → down. qkv, attention and the rest are timed separately in each step. The cache is held at a fixed n (no append; the 11/18 append cost was < 0.1 ms).
- Configs: u32 weights × caches {a8, k8v4, khv4h4}, plus int8 weights + a8 as the control. Grid: ctx ∈ {1024, 2048, 4096, 8192} × B ∈ {1, 8}, plus ctx 16384 at B = 1. D 2560, F 6912, 20 q / 5 KV heads, HD 128. TK ∈ {1024, 2048} is picked per config by a 5-call pre-timing.
- Timing: 5 rounds, with the config order rotated each round. Each round has 2 warmup + 6 timed steps, so n = 30 steps per config. Tables give the median (min–max). Two runs: SEED 11210555 (new) and SEED 11210255 (the 02:55 seed, which regenerates its KV data). The synthetic KV generator is the 11/19–11/21 one (6 planted ×6 outlier K channels per head).
- Error: layer output delta cos vs the same step with exact f32 softmax attention over the f32 history ('iq.ref'), min over streams.
- Python 3 + numpy, OPENBLAS_NUM_THREADS=1, 8 cores. No network, nothing bound, no installs.

Command:

    cd /workspace/bench0555_1121
    OPENBLAS_NUM_THREADS=1 SEED=11210555 python3 dstep.py results_run1.json > run1.txt
    OPENBLAS_NUM_THREADS=1 SEED=11210255 python3 dstep.py results_run2.json > run2.txt
    python3 cross.py > cross.txt

## Results

### Run 1 (SEED 11210555)

| cfg | step ms med (min–max) | qkv ms | attn ms | rest ms | attn share | tok/s | TK | KV MiB | total MiB | layer cos |
|---|---|---|---|---|---|---|---|---|---|---|
| u32+a8_ctx1024_B1 | 21.35 (20.17–24.96) | 3.20 | 1.10 | 16.95 | 0.052 | 46.8 | 2048 | 1.33 | 17.98 | 0.999888 |
| u32+k8v4_ctx1024_B1 | 21.91 (20.30–23.90) | 3.37 | 1.16 | 17.48 | 0.054 | 45.6 | 2048 | 1.33 | 17.98 | 0.999888 |
| u32+khv4h4_ctx1024_B1 | 21.82 (20.29–25.99) | 3.28 | 1.38 | 17.27 | 0.062 | 45.8 | 2048 | 1.31 | 17.96 | 0.999996 |
| int8+a8_ctx1024_B1 | 15.99 (14.55–18.05) | 2.51 | 1.21 | 12.11 | 0.076 | 62.5 | 2048 | 1.33 | 67.67 | 0.999645 |
| u32+a8_ctx1024_B8 | 21.06 (18.07–24.18) | 2.24 | 6.12 | 12.49 | 0.288 | 379.8 | 2048 | 10.62 | 27.27 | 0.999743 |
| u32+k8v4_ctx1024_B8 | 21.30 (18.67–24.87) | 2.31 | 6.08 | 12.54 | 0.293 | 375.5 | 2048 | 10.62 | 27.27 | 0.999743 |
| u32+khv4h4_ctx1024_B8 | 20.79 (19.15–28.72) | 2.13 | 6.85 | 11.81 | 0.328 | 384.7 | 2048 | 10.47 | 27.12 | 0.999993 |
| int8+a8_ctx1024_B8 | 15.21 (14.36–16.88) | 1.45 | 5.78 | 7.95 | 0.380 | 526.0 | 1024 | 10.62 | 76.96 | 0.998991 |
| u32+a8_ctx2048_B1 | 22.27 (20.83–24.23) | 3.31 | 1.74 | 17.02 | 0.076 | 44.9 | 1024 | 2.66 | 19.31 | 0.999488 |
| u32+k8v4_ctx2048_B1 | 22.91 (20.89–26.44) | 3.30 | 1.84 | 17.65 | 0.081 | 43.6 | 2048 | 2.38 | 19.03 | 0.999301 |
| u32+khv4h4_ctx2048_B1 | 23.65 (21.60–27.81) | 3.35 | 2.22 | 17.88 | 0.092 | 42.3 | 1024 | 2.32 | 18.97 | 0.999749 |
| int8+a8_ctx2048_B1 | 16.74 (15.42–23.60) | 2.58 | 1.69 | 12.33 | 0.101 | 59.7 | 2048 | 2.66 | 68.99 | 0.999548 |
| u32+a8_ctx2048_B8 | 22.65 (21.33–28.53) | 2.35 | 7.31 | 12.91 | 0.319 | 353.2 | 2048 | 21.25 | 37.90 | 0.999488 |
| u32+k8v4_ctx2048_B8 | 26.05 (23.83–29.96) | 2.22 | 10.83 | 12.88 | 0.423 | 307.1 | 2048 | 19.06 | 35.71 | 0.999301 |
| u32+khv4h4_ctx2048_B8 | 28.88 (26.54–42.47) | 2.45 | 12.94 | 13.08 | 0.458 | 277.0 | 2048 | 18.59 | 35.24 | 0.999662 |
| int8+a8_ctx2048_B8 | 16.84 (15.39–22.30) | 1.51 | 7.21 | 8.11 | 0.422 | 475.2 | 2048 | 21.25 | 87.59 | 0.999486 |
| u32+a8_ctx4096_B1 | 25.45 (22.17–27.92) | 3.54 | 2.60 | 18.99 | 0.104 | 39.3 | 2048 | 5.31 | 21.96 | 0.999996 |
| u32+k8v4_ctx4096_B1 | 25.15 (23.20–28.35) | 3.47 | 3.24 | 18.48 | 0.129 | 39.8 | 2048 | 4.49 | 21.14 | 0.999945 |
| u32+khv4h4_ctx4096_B1 | 26.11 (23.68–39.67) | 3.46 | 3.76 | 18.76 | 0.144 | 38.3 | 2048 | 4.36 | 21.00 | 0.999948 |
| int8+a8_ctx4096_B1 | 17.81 (16.30–21.21) | 2.49 | 2.56 | 12.59 | 0.145 | 56.1 | 2048 | 5.31 | 71.65 | 0.999984 |
| u32+a8_ctx4096_B8 | 28.55 (26.21–34.64) | 2.37 | 12.98 | 13.18 | 0.453 | 280.2 | 2048 | 42.50 | 59.15 | 0.999968 |
| u32+k8v4_ctx4096_B8 | 33.70 (29.90–42.51) | 2.68 | 17.57 | 14.03 | 0.512 | 237.4 | 2048 | 35.94 | 52.59 | 0.999784 |
| u32+khv4h4_ctx4096_B8 | 38.82 (34.03–48.64) | 2.54 | 21.75 | 14.77 | 0.557 | 206.1 | 2048 | 34.84 | 51.49 | 0.999801 |
| int8+a8_ctx4096_B8 | 23.36 (20.31–29.27) | 1.66 | 12.88 | 8.50 | 0.560 | 342.5 | 2048 | 42.50 | 108.84 | 0.999655 |
| u32+a8_ctx8192_B1 | 25.86 (23.48–29.38) | 3.38 | 4.13 | 18.20 | 0.158 | 38.7 | 2048 | 10.62 | 27.27 | 0.999998 |
| u32+k8v4_ctx8192_B1 | 27.05 (24.31–30.80) | 3.44 | 5.18 | 18.53 | 0.190 | 37.0 | 2048 | 8.71 | 25.36 | 0.999963 |
| u32+khv4h4_ctx8192_B1 | 27.32 (24.81–30.31) | 3.37 | 5.69 | 18.09 | 0.204 | 36.6 | 2048 | 8.42 | 25.07 | 0.999963 |
| int8+a8_ctx8192_B1 | 19.12 (17.88–22.08) | 2.63 | 4.04 | 12.40 | 0.209 | 52.3 | 2048 | 10.62 | 76.96 | 0.999940 |
| u32+a8_ctx8192_B8 | 40.35 (34.16–50.61) | 2.42 | 23.26 | 14.45 | 0.579 | 198.3 | 2048 | 85.00 | 101.65 | 0.999982 |
| u32+k8v4_ctx8192_B8 | 44.33 (40.15–52.35) | 2.49 | 27.75 | 13.57 | 0.636 | 180.5 | 2048 | 69.69 | 86.34 | 0.999834 |
| u32+khv4h4_ctx8192_B8 | 49.12 (45.64–57.85) | 2.21 | 33.50 | 13.54 | 0.682 | 162.9 | 2048 | 67.34 | 83.99 | 0.999859 |
| int8+a8_ctx8192_B8 | 32.62 (30.40–40.57) | 1.69 | 22.32 | 8.43 | 0.689 | 245.3 | 2048 | 85.00 | 151.34 | 0.999806 |
| u32+a8_ctx16384_B1 | 30.49 (27.87–33.72) | 3.39 | 7.76 | 19.15 | 0.257 | 32.8 | 2048 | 21.25 | 37.90 | 0.999983 |
| u32+k8v4_ctx16384_B1 | 33.34 (30.39–37.37) | 3.44 | 10.96 | 19.02 | 0.323 | 30.0 | 1024 | 17.15 | 33.80 | 0.999915 |
| u32+khv4h4_ctx16384_B1 | 33.45 (29.74–37.03) | 3.53 | 10.67 | 19.60 | 0.318 | 29.9 | 2048 | 16.54 | 33.19 | 0.999935 |
| int8+a8_ctx16384_B1 | 23.61 (20.07–27.27) | 2.53 | 7.22 | 13.18 | 0.319 | 42.4 | 2048 | 21.25 | 87.59 | 0.999877 |

### Run 2 (SEED 11210255)

| cfg | step ms med (min–max) | qkv ms | attn ms | rest ms | attn share | tok/s | TK | KV MiB | total MiB | layer cos |
|---|---|---|---|---|---|---|---|---|---|---|
| u32+a8_ctx1024_B1 | 25.14 (22.22–27.29) | 3.61 | 1.31 | 20.07 | 0.053 | 39.8 | 2048 | 1.33 | 17.98 | 0.999981 |
| u32+k8v4_ctx1024_B1 | 24.44 (22.03–27.50) | 3.56 | 1.34 | 19.46 | 0.055 | 40.9 | 2048 | 1.33 | 17.98 | 0.999981 |
| u32+khv4h4_ctx1024_B1 | 24.94 (21.93–30.83) | 3.60 | 1.54 | 19.71 | 0.062 | 40.1 | 1024 | 1.31 | 17.96 | 0.999997 |
| int8+a8_ctx1024_B1 | 17.05 (15.38–19.32) | 2.60 | 1.30 | 13.01 | 0.076 | 58.7 | 1024 | 1.33 | 67.67 | 0.999938 |
| u32+a8_ctx1024_B8 | 26.60 (22.47–31.04) | 3.02 | 7.28 | 16.44 | 0.271 | 300.8 | 2048 | 10.62 | 27.27 | 0.999909 |
| u32+k8v4_ctx1024_B8 | 23.79 (21.96–30.83) | 2.61 | 6.75 | 14.68 | 0.278 | 336.3 | 2048 | 10.62 | 27.27 | 0.999909 |
| u32+khv4h4_ctx1024_B8 | 25.78 (23.30–33.04) | 2.71 | 7.76 | 15.25 | 0.301 | 310.3 | 1024 | 10.47 | 27.12 | 0.999991 |
| int8+a8_ctx1024_B8 | 18.26 (16.24–23.22) | 1.71 | 6.77 | 9.91 | 0.364 | 438.1 | 2048 | 10.62 | 76.96 | 0.999782 |
| u32+a8_ctx2048_B1 | 25.63 (23.24–29.20) | 3.65 | 1.90 | 19.93 | 0.073 | 39.0 | 1024 | 2.66 | 19.31 | 0.999977 |
| u32+k8v4_ctx2048_B1 | 26.16 (23.09–28.15) | 3.81 | 2.07 | 20.27 | 0.081 | 38.2 | 2048 | 2.38 | 19.03 | 0.999898 |
| u32+khv4h4_ctx2048_B1 | 26.26 (24.02–30.15) | 3.74 | 2.31 | 20.08 | 0.089 | 38.1 | 2048 | 2.32 | 18.97 | 0.999919 |
| int8+a8_ctx2048_B1 | 18.13 (16.64–22.89) | 2.84 | 1.84 | 13.37 | 0.102 | 55.2 | 2048 | 2.66 | 68.99 | 0.999781 |
| u32+a8_ctx2048_B8 | 25.42 (22.31–27.23) | 2.43 | 7.93 | 14.62 | 0.318 | 314.7 | 2048 | 21.25 | 37.90 | 0.999900 |
| u32+k8v4_ctx2048_B8 | 29.49 (26.21–35.04) | 2.54 | 11.87 | 15.45 | 0.398 | 271.3 | 2048 | 19.06 | 35.71 | 0.999778 |
| u32+khv4h4_ctx2048_B8 | 32.48 (29.24–43.38) | 2.66 | 13.97 | 15.49 | 0.435 | 246.3 | 1024 | 18.59 | 35.24 | 0.999862 |
| int8+a8_ctx2048_B8 | 18.96 (16.70–21.35) | 1.72 | 7.88 | 9.21 | 0.418 | 421.9 | 2048 | 21.25 | 87.59 | 0.999781 |
| u32+a8_ctx4096_B1 | 27.23 (24.67–31.51) | 4.08 | 2.78 | 20.46 | 0.102 | 36.7 | 2048 | 5.31 | 21.96 | 0.999990 |
| u32+k8v4_ctx4096_B1 | 29.39 (25.77–37.21) | 4.07 | 3.51 | 21.51 | 0.119 | 34.0 | 2048 | 4.49 | 21.14 | 0.999912 |
| u32+khv4h4_ctx4096_B1 | 29.21 (25.48–35.33) | 4.15 | 3.91 | 21.18 | 0.139 | 34.2 | 2048 | 4.36 | 21.00 | 0.999921 |
| int8+a8_ctx4096_B1 | 22.06 (17.25–26.86) | 3.18 | 3.31 | 15.12 | 0.149 | 45.3 | 2048 | 5.31 | 71.65 | 0.999893 |
| u32+a8_ctx4096_B8 | 31.47 (27.90–43.17) | 2.70 | 13.09 | 15.63 | 0.413 | 254.2 | 2048 | 42.50 | 59.15 | 0.999914 |
| u32+k8v4_ctx4096_B8 | 38.12 (32.44–53.01) | 2.84 | 19.12 | 16.33 | 0.488 | 209.9 | 2048 | 35.94 | 52.59 | 0.999557 |
| u32+khv4h4_ctx4096_B8 | 39.37 (35.71–50.61) | 2.61 | 21.56 | 15.39 | 0.543 | 203.2 | 2048 | 34.84 | 51.49 | 0.999647 |
| int8+a8_ctx4096_B8 | 24.88 (21.94–33.11) | 1.72 | 12.97 | 9.95 | 0.526 | 321.6 | 2048 | 42.50 | 108.84 | 0.999703 |
| u32+a8_ctx8192_B1 | 28.62 (25.16–31.88) | 3.54 | 4.54 | 20.19 | 0.163 | 34.9 | 2048 | 10.62 | 27.27 | 0.999997 |
| u32+k8v4_ctx8192_B1 | 28.45 (26.38–33.69) | 3.39 | 5.16 | 19.55 | 0.186 | 35.2 | 2048 | 8.71 | 25.36 | 0.999951 |
| u32+khv4h4_ctx8192_B1 | 30.09 (26.85–39.82) | 3.52 | 6.01 | 20.58 | 0.203 | 33.2 | 2048 | 8.42 | 25.07 | 0.999953 |
| int8+a8_ctx8192_B1 | 19.91 (17.90–28.78) | 2.38 | 4.61 | 12.89 | 0.227 | 50.2 | 2048 | 10.62 | 76.96 | 0.999981 |
| u32+a8_ctx8192_B8 | 45.17 (41.17–58.78) | 2.73 | 26.45 | 15.90 | 0.585 | 177.1 | 2048 | 85.00 | 101.65 | 0.999702 |
| u32+k8v4_ctx8192_B8 | 52.03 (45.46–63.10) | 2.82 | 32.70 | 15.79 | 0.636 | 153.8 | 2048 | 69.69 | 86.34 | 0.998739 |
| u32+khv4h4_ctx8192_B8 | 64.02 (54.10–93.59) | 2.95 | 43.52 | 17.02 | 0.676 | 125.0 | 2048 | 67.34 | 83.99 | 0.999205 |
| int8+a8_ctx8192_B8 | 41.15 (35.22–43.95) | 2.02 | 27.14 | 11.35 | 0.665 | 194.4 | 2048 | 85.00 | 151.34 | 0.999483 |
| u32+a8_ctx16384_B1 | 27.84 (25.73–37.72) | 3.22 | 6.92 | 17.57 | 0.248 | 35.9 | 2048 | 21.25 | 37.90 | 0.999990 |
| u32+k8v4_ctx16384_B1 | 29.46 (27.28–35.55) | 3.17 | 8.34 | 17.68 | 0.282 | 33.9 | 2048 | 17.15 | 33.80 | 0.999935 |
| u32+khv4h4_ctx16384_B1 | 31.27 (29.31–49.56) | 3.29 | 9.56 | 18.66 | 0.306 | 32.0 | 2048 | 16.54 | 33.19 | 0.999946 |
| int8+a8_ctx16384_B1 | 20.54 (18.75–27.08) | 2.24 | 6.84 | 11.56 | 0.334 | 48.7 | 2048 | 21.25 | 87.59 | 0.999982 |

### Ratios (> 1 means the first format is faster or smaller)

| comparison | run1 attn | run1 step | run2 attn | run2 step | total bytes |
|---|---|---|---|---|---|
| khv4h4_vs_a8_ctx1024_B1 | 0.800x | 0.978x | 0.855x | 1.008x | 1.001x |
| khv4h4_vs_k8v4_ctx1024_B1 | 0.843x | 1.004x | 0.871x | 0.980x | 1.001x |
| u32khv4h4_vs_int8a8_ctx1024_B1 | – | 0.733x | – | 0.684x | 3.768x |
| khv4h4_vs_a8_ctx1024_B8 | 0.892x | 1.013x | 0.938x | 1.032x | 1.006x |
| khv4h4_vs_k8v4_ctx1024_B8 | 0.888x | 1.025x | 0.870x | 0.923x | 1.006x |
| u32khv4h4_vs_int8a8_ctx1024_B8 | – | 0.731x | – | 0.708x | 2.838x |
| khv4h4_vs_a8_ctx2048_B1 | 0.784x | 0.942x | 0.822x | 0.976x | 1.017x |
| khv4h4_vs_k8v4_ctx2048_B1 | 0.826x | 0.969x | 0.895x | 0.996x | 1.003x |
| u32khv4h4_vs_int8a8_ctx2048_B1 | – | 0.708x | – | 0.690x | 3.636x |
| khv4h4_vs_a8_ctx2048_B8 | 0.565x | 0.784x | 0.568x | 0.783x | 1.075x |
| khv4h4_vs_k8v4_ctx2048_B8 | 0.837x | 0.902x | 0.850x | 0.908x | 1.013x |
| u32khv4h4_vs_int8a8_ctx2048_B8 | – | 0.583x | – | 0.584x | 2.485x |
| khv4h4_vs_a8_ctx4096_B1 | 0.692x | 0.975x | 0.710x | 0.932x | 1.046x |
| khv4h4_vs_k8v4_ctx4096_B1 | 0.860x | 0.963x | 0.898x | 1.006x | 1.007x |
| u32khv4h4_vs_int8a8_ctx4096_B1 | – | 0.682x | – | 0.755x | 3.411x |
| khv4h4_vs_a8_ctx4096_B8 | 0.597x | 0.735x | 0.607x | 0.799x | 1.149x |
| khv4h4_vs_k8v4_ctx4096_B8 | 0.808x | 0.868x | 0.887x | 0.968x | 1.021x |
| u32khv4h4_vs_int8a8_ctx4096_B8 | – | 0.602x | – | 0.632x | 2.114x |
| khv4h4_vs_a8_ctx8192_B1 | 0.725x | 0.947x | 0.755x | 0.951x | 1.088x |
| khv4h4_vs_k8v4_ctx8192_B1 | 0.911x | 0.990x | 0.858x | 0.946x | 1.012x |
| u32khv4h4_vs_int8a8_ctx8192_B1 | – | 0.700x | – | 0.662x | 3.070x |
| khv4h4_vs_a8_ctx8192_B8 | 0.694x | 0.821x | 0.608x | 0.706x | 1.210x |
| khv4h4_vs_k8v4_ctx8192_B8 | 0.828x | 0.902x | 0.751x | 0.813x | 1.028x |
| u32khv4h4_vs_int8a8_ctx8192_B8 | – | 0.664x | – | 0.643x | 1.802x |
| khv4h4_vs_a8_ctx16384_B1 | 0.727x | 0.911x | 0.723x | 0.890x | 1.142x |
| khv4h4_vs_k8v4_ctx16384_B1 | 1.027x | 0.997x | 0.872x | 0.942x | 1.018x |
| u32khv4h4_vs_int8a8_ctx16384_B1 | – | 0.706x | – | 0.657x | 2.639x |

### Attention crossover (linear fit, attn ms = a + b·ctx; weights = median qkv + rest)

    B1 u32+a8      run1 attn=0.77+0.424ms/1k  weights 21.58 ms  50% at ctx 49,017  25% at ctx 15,125
    B1 u32+a8      run2 attn=1.19+0.361ms/1k  weights 23.68 ms  50% at ctx 62,241  25% at ctx 18,543
    B1 u32+k8v4    run1 attn=0.48+0.629ms/1k  weights 21.95 ms  50% at ctx 34,135  25% at ctx 10,866
    B1 u32+k8v4    run2 attn=1.27+0.444ms/1k  weights 23.03 ms  50% at ctx 49,042  25% at ctx 14,447
    B1 u32+khv4h4  run1 attn=1.00+0.591ms/1k  weights 21.46 ms  50% at ctx 34,662  25% at ctx 10,430
    B1 u32+khv4h4  run2 attn=1.41+0.513ms/1k  weights 23.83 ms  50% at ctx 43,669  25% at ctx 12,730
    B1 int8+a8     run1 attn=0.89+0.387ms/1k  weights 15.03 ms  50% at ctx 36,546  25% at ctx 10,649
    B1 int8+a8     run2 attn=1.35+0.351ms/1k  weights 15.61 ms  50% at ctx 40,568  25% at ctx 10,967
    B8 u32+a8      run1 attn=2.96+2.462ms/1k  weights 15.41 ms  50% at ctx 5,054  25% at ctx 883
    B8 u32+a8      run2 attn=3.02+2.779ms/1k  weights 18.48 ms  50% at ctx 5,565  25% at ctx 1,131
    B8 u32+k8v4    run1 attn=4.26+2.943ms/1k  weights 15.58 ms  50% at ctx 3,849  25% at ctx 319
    B8 u32+k8v4    run2 attn=3.99+3.547ms/1k  weights 18.30 ms  50% at ctx 4,034  25% at ctx 595
    B8 u32+khv4h4  run1 attn=4.89+3.611ms/1k  weights 15.64 ms  50% at ctx 2,975  25% at ctx 88
    B8 u32+khv4h4  run2 attn=2.85+4.910ms/1k  weights 18.08 ms  50% at ctx 3,102  25% at ctx 647
    B8 int8+a8     run1 attn=2.97+2.363ms/1k  weights 9.87 ms  50% at ctx 2,918  25% at ctx 133
    B8 int8+a8     run2 attn=2.45+2.927ms/1k  weights 11.64 ms  50% at ctx 3,141  25% at ctx 489

## Verdict

**not-viable.**
- At B = 8, attention is ≥ 30% of the step even at ctx 1024 and 68% at ctx 8192. khv4h4's slower attention therefore shows up almost in full: 0.81–0.97x of K8V4 tok/s for a 1.3–2.8% byte saving.
- At B = 1 the cost disappears (0.96–1.00x), but so does the reason to use the format.
- u32 weights + khv4h4 keep the 1.80x total-memory win vs int8 + A8 at B = 8 / ctx 8192, at 0.64–0.66x of int8 tok/s. That is unchanged from 11/18–11/21, not a new gain.
- Recommendation for i2sd: K8V4 as the default cache. Speed work should target attention, not weight GEMVs, once B·ctx ≳ 3k tokens per stream at B = 8.

## Pair files this wave

No Trit Pack or Handshake pair reports reached chat for this wave. The quant derivation lane is paused, and the JuniorHome docs/ROADMAP.md and JuniorLLM docs/BETA_TO_OS.md queues are still exhausted. The last 6 h of JuniorLLM show only my 02:55 note (536997c) and an unattributed 02:55 rails/pathfree push (fe254fb). This note touches neither path.

## Script (dstep.py)

    #!/usr/bin/env python3
    """Full bitnetd decode step with compressed KV caches, 05:55 11/21. B decode streams share one 2-bit u32-LUT weight pass
    (par.fmm from /workspace/bench1455_1118, T=256, W=8) per step: x -> rms -> fused qkv -> attention over each stream's cache
    (iq.py 02:55 11/21 attention: online softmax, 8-thread pool over (stream, KV head), best TK per format) -> o -> rms ->
    gate|up -> silu -> down. Caches: a8 (all A8 g64), k8v4, khv4h4 (4 hot K ch f16 + A8 g64 K f16 scales + A4 g32 old V,
    A8 1024 window). Control: int8-resident weights (par.fmm8) + a8 cache. Cache is held at fixed n (no append; the 11/18
    step's append was < 0.1 ms). Per step we time qkv, attention and the rest separately.
    Question: share of step wall time in attention vs 2-bit weight GEMVs, end-to-end cost of khv4h4's slower attention,
    and the ctx where attention crosses 50% of the step.
    Usage: OPENBLAS_NUM_THREADS=1 SEED=... python3 dstep.py out.json"""
    import json, sys, time, statistics, os, types
    import numpy as np
    sys.path.insert(0, "/workspace/bench1455_1118")
    import base, par
    from base import rms, silu, D
    src = open("/workspace/bench0255_1121/iq.py").read().rstrip()
    assert src.endswith("main()"); iq = types.ModuleType("iq"); exec(compile(src[:-6], "iq.py", "exec"), iq.__dict__)
    KVH, G, HD, SC = iq.KVH, iq.G, iq.HD, iq.SC
    TU, W = 256, 8
    FMTS = ("a8", "k8v4", "khv4h4")
    RUNS = [("u32", f) for f in FMTS] + [("int8", "a8")]
    GRID = [(n, B) for n in (1024, 2048, 4096, 8192) for B in (1, 8)] + [(16384, 1)]
    ROUNDS, STEPS, WARM = 5, 6, 2
    base.build(); WB = {"u32": par.prep(), "int8": par.prep8()}
    for nm in base.PW:
        for f in ("Wd", "Wt", "wm", "ws", "wm8", "ws8", "PT", "q8"): base.PW[nm].pop(f, None)
    MM = {"u32": par.fmm, "int8": par.fmm8}
    def step(mm, x, af):
        t0 = time.perf_counter(); q = mm("qkv", rms(x), TU, W)[0]
        t1 = time.perf_counter(); att = af(q.reshape(x.shape[0], KVH, G, HD) * SC).reshape(x.shape[0], D)
        t2 = time.perf_counter(); x2 = x + mm("o", att, TU, W)[0]
        g, u = mm("gu", rms(x2), TU, W); y = x2 + mm("down", (silu(g) * u).astype(np.float32), TU, W)[0]
        t3 = time.perf_counter(); return y, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)
    def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    def main():
        rng = iq.rng; out = {"seed": iq.SEED, "weight_bytes": WB, "cfg": {}}
        print(f"seed {iq.SEED} | weights u32 {WB['u32']/2**20:.2f} MiB int8 {WB['int8']/2**20:.2f} MiB")
        print(f"{'cfg':26s} {'step ms med (min-max)':>24s} {'qkv':>6s} {'attn':>7s} {'rest':>6s} share  tok/s  TK   KV MiB  tot MiB  layer cos")
        XS = rng.standard_normal((STEPS, 8, D)).astype(np.float32)
        for n in sorted({g[0] for g in GRID}):
            Bmax = max(B for m, B in GRID if m == n)
            kvs8 = [iq.KV(*iq.kvhist(n)) for _ in range(Bmax)]
            for B in [B for m, B in GRID if m == n]:
                kvs = kvs8[:B]; x0 = XS[0, :B]
                refaf = lambda qq: np.stack([np.stack([iq.ref(kvs[s], qq[s, h], h) for h in range(KVH)]) for s in range(B)])
                yref = {wk: step(MM[wk], x0, refaf)[0] for wk in MM}
                cfgs = {}
                for wk, fmt in RUNS:
                    qq = (MM[wk]("qkv", rms(x0), TU, W)[0].reshape(B, KVH, G, HD) * SC)
                    tk = min((1024, 2048), key=lambda T: statistics.median([(lambda t0: (iq.attn(kvs, qq, fmt, T), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(5)]))
                    af = lambda qq, fmt=fmt, tk=tk: iq.attn(kvs, qq, fmt, tk)
                    y = step(MM[wk], x0, af)[0]
                    cfgs[(wk, fmt)] = dict(TK=tk, af=af, t=[], lc=min(cos(y[s] - x0[s], yref[wk][s] - x0[s]) for s in range(B)),
                                           kv=B * kvs[0].bytes(fmt) / 2**20)
                order = list(cfgs)
                for rd in range(ROUNDS):
                    for kk in order[rd % len(order):] + order[:rd % len(order)]:
                        c = cfgs[kk]
                        for i in range(WARM + STEPS):
                            _, t = step(MM[kk[0]], XS[i % STEPS, :B], c["af"])
                            if i >= WARM: c["t"].append([v * 1e3 for v in t])
                for (wk, fmt), c in cfgs.items():
                    T = np.array(c["t"]); med = [float(np.median(T[:, j])) for j in range(4)]
                    key = f"{wk}+{fmt}_ctx{n}_B{B}"
                    d = dict(B=B, ctx=n, TK=c["TK"], qkv=med[0], attn=med[1], rest=med[2], step=med[3], step_lo=float(T[:, 3].min()), step_hi=float(T[:, 3].max()),
                             share=float(np.median(T[:, 1] / T[:, 3])), tok_s=B * 1e3 / med[3], kv_mib=c["kv"], tot_mib=WB[wk] / 2**20 + c["kv"], layer_cos=c["lc"], nsteps=len(T))
                    out["cfg"][key] = d
                    print(f"{key:26s} {d['step']:8.2f} ({d['step_lo']:.2f}-{d['step_hi']:.2f}) {d['qkv']:6.2f} {d['attn']:7.2f} {d['rest']:6.2f} {d['share']:.3f} {d['tok_s']:6.1f} {d['TK']:5d} {d['kv_mib']:7.2f} {d['tot_mib']:8.2f}  {d['layer_cos']:.6f}", flush=True)
                C = lambda f, wk="u32": out["cfg"][f"{wk}+{f}_ctx{n}_B{B}"]
                for f in ("khv4h4",):
                    for r in ("a8", "k8v4"):
                        out.setdefault("cmp", {})[f"{f}_vs_{r}_ctx{n}_B{B}"] = dict(attn=C(r)["attn"] / C(f)["attn"], step=C(r)["step"] / C(f)["step"], bytes=C(r)["tot_mib"] / C(f)["tot_mib"])
                out["cmp"][f"u32khv4h4_vs_int8a8_ctx{n}_B{B}"] = dict(step=C("a8", "int8")["step"] / C("khv4h4")["step"], bytes=C("a8", "int8")["tot_mib"] / C("khv4h4")["tot_mib"])
                for k2, v in out["cmp"].items():
                    if k2.endswith(f"_ctx{n}_B{B}"): print(f"   {k2:34s} " + "  ".join(f"{a} {b:.3f}x" for a, b in v.items()))
            del kvs8
        json.dump(out, open(sys.argv[1], "w"), indent=1)
    main()

cross.py:

    import json, numpy as np
    R=[json.load(open(f"/workspace/bench0555_1121/results_run{i}.json")) for i in (1,2)]
    for B in (1,8):
      for k in ("u32+a8","u32+k8v4","u32+khv4h4","int8+a8"):
        for i,r in enumerate(R):
          rows=[v for kk,v in r["cfg"].items() if kk.startswith(k+"_") and v["B"]==B]
          n=np.array([v["ctx"] for v in rows],float); a=np.array([v["attn"] for v in rows]); w=np.array([v["qkv"]+v["rest"] for v in rows])
          b,c=np.polyfit(n,a,1); W=float(np.median(w)); x50=(W-c)/b; x25=(W/3-c)/b
          print(f"B{B} {k:11s} run{i+1} attn={c:.2f}+{b*1e3:.3f}ms/1k  weights {W:.2f} ms  50% at ctx {x50:,.0f}  25% at ctx {x25:,.0f}")

## Next question

Attention is the decode bottleneck at B = 8 past ctx ≈ 3k, and all-A8 attention runs at 1.2–1.7x of the compressed formats. So: at B = 8 / ctx 8192, does a **split-K (flash-decoding) partition of the 8192-token history across the 8 workers per stream** (merging partial (m, l, acc) per head), instead of one task per (stream, KV head), raise attention throughput ≥ 1.5x for K8V4 and khv4h4? 40 tasks on 8 threads leave a tail, and B = 1 uses only 5 of 8 threads. And what does that do to the end-to-end step share?

close: e8472e48c358744b
