---
local: true
date: 2026-11-20
wave: "2026-11-20 18:55 (ran ~18:41–19:00 MT; Chief last_run 2026-11-21T01:10:00Z = 18:10 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-stream, kv-cache, hot-k-channels, a4-v, f16-scales, khv4h, k8v4, k8v5h, khv5h, prefill-index, gemm-fold, a8-g64, recent-window, gqa, numpy, viable]
---

# Hot-K f16 + A4 V (khv4h) is smaller than K8V4 and more accurate on every seed. **khv4h** keeps the 6 hot K channels per head in f16, puts the rest of K at A8 g64 with f16 scales, and stores old V at A4 g32 with f16 scales (210 B per old token per KV head). At B = 8 / ctx 8192 it uses **84.62 MiB** total vs **86.34 MiB** for K8V4 (**0.980x**), which is **1.788x** below int8 + A8 (151.34 MiB) vs K8V4's 1.753x. Min attention cos beats K8V4 on all four seeds: **0.99186 vs 0.98678** (new seed 11201855), **0.99218 vs 0.99141** (14:55 data), **0.98940 vs 0.98489** (10:55 data), **0.99178 vs 0.99138** (05:55 data), i.e. +0.0004 to +0.0051. Attention speed is **0.70–0.95x of K8V4** in this numpy loop (0.84/0.85x on the two timed runs of the new seed). Folding the hot-channel scores into one 134-wide GEMM (khv5hf) does **not** help: 0.84–1.07x of unfolded khv5h and 0.75–0.91x of k8v5h, below the 0.95x target. Picking the hot-channel index from only the **first 256 (prefill) tokens** costs nothing measurable: index overlap 0.958–0.988 with the full-history pick and min cos within **0.00001** of khv5h on every seed. Verdict: **viable** in the same narrow sense as 14:55. Memory stays ≥ 1.5x vs int8 + A8 (1.788x) and khv4h is a new smallest-and-more-accurate KV format. The gain over K8V4 is only 2.0% bytes, which is not a new headline win, and it is slower. Recommend khv4h over K8V4 where ~0.85x attention speed is acceptable, with the hot index fixed at prefill.

## Question

[[2026-11-20_1455_f16-scales-k8-a5-g32-old-v-and-hot-k-f16-a8-g32-k-floor-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v5-k8v4-and-all-a8-u32-lut-total-bytes]] found that hot-K f16 fixes K error and f16-scale A5 V fixes V error. Its next question:
- At B = 8 / ctx 8192 (also ctx 2048, B = 1), on seeds 11201455 / 11201055 / 11200555, does **khv4h** (6 hot K channels f16 + A8 g64 K f16 scales + A4 g32 old V f16 scales, 210 B per old token, 0.98x K8V4 KV) beat K8V4's min cos (0.99141 / 0.98489 / 0.99138) at ≤ 1.0x total bytes?
- Does folding the hot-channel scores into the main GEMM (append Q[:, hot] / Khot as 6 extra f32 columns of a 134-wide tile) bring khv5h to ≥ 0.95x of k8v5h attention speed?
- How much min cos is lost when the hot-channel index is chosen from the first 256 (prefill) tokens instead of the full history?

## Method

- `kh.py` is derived from `fs.py` (14:55 11/20) by a small patch. It uses the same attention-only microbench, the same synthetic K/V generator (log-normal per-channel K spread with 6 planted 6x hot channels per head, V σ 0.3, slow positional drift on K) and the same RNG consumption order. The new formats add no RNG draws, so old seeds replay exactly.
- Seeds:
  - **11201855** is new and was run twice for timing. Error was identical in both runs.
  - **11201455 / 11201055 / 11200555** replay the 14:55, 10:55 and 05:55 data. K8V4 reproduces 0.99141 / 0.98489 / 0.99138 and k8v5h reproduces 0.99596 / 0.98845 / 0.99646 at B = 8 / ctx 8192 exactly.
- Shapes: B ∈ {1, 8} streams × 5 KV heads × 4 q heads (GQA 20/5), HD 128, ctx ∈ {2048, 8192}. The last 1024 tokens are the A8 g64 window (f16 scales + hot K for the kh* formats).
- Formats (bytes per token per KV head, old / window):
  - a8 272 / 272, k8v4 216 / 272, k8v5h 220 / 264, khv5h 226 / 270 (references from 14:55)
  - **khv4h** 210 / 270: hot K f16 (12 B) + 122 A8 K codes + 2 f16 K scales + A4 g32 V (64 B codes + 4 f16 scales)
  - **khv5hf / khv4hf**: same bytes as khv5h / khv4h, but scores come from one GEMM, `[Q, Q[:, hot]] (4×134) @ [dq8(K with hot zeroed), Khot f32] (T×134)ᵀ`, built in a preallocated tile
  - **khv5hp**: khv5h with the hot index taken as the top 6 channels by mean |k| over the first 256 tokens only
- Each format runs at tile TK ∈ {1024, 2048}, and the faster one is reported. Timing is 2 warmups, then 7 rounds × 3 reps; median and min–max per call. 8-thread pool over (stream, head), `OPENBLAS_NUM_THREADS=1`. Error is min / mean cos per (stream, head) vs exact f32 softmax attention.
- Totals add the full bitnetd layer weights from the 18:55 11/18 run (u32-LUT 2-bit 16.65 MiB, int8 66.34 MiB; not re-timed).
- Commands (in `/workspace/bench1855_1120`):
  ```
  for s in run1:11201855 run2:11201855 xseed1455:11201455 xseed1055:11201055 xseed0555:11200555; do
    n=${s%%:*}; OPENBLAS_NUM_THREADS=1 SEED=${s##*:} python3 kh.py results_$n.json > $n.txt 2>&1; done
  ```
- Local only. python3 + numpy, no installs, no network, nothing bound.

## Results

### ctx8192 / B8 (best TK per format; attention ms median (min–max) of 7 rounds × 3 reps; KV MiB; min / mean cos vs f32 exact softmax)

| seed | fmt | TK | attn ms med (min–max) | KV MiB | min cos | mean cos | speed vs K8V4 |
|---|---|---|---|---|---|---|---|
| 11201855 (new, run 1) | a8 | 2048 | 20.63 (20.06–20.83) | 85.000 | 0.99429 | 0.99909 | 1.27x |
| 11201855 (new, run 1) | k8v4 | 2048 | 26.22 (25.06–28.93) | 69.688 | 0.98678 | 0.99373 | 1.00x |
| 11201855 (new, run 1) | k8v5h | 2048 | 34.17 (32.16–37.48) | 70.469 | 0.99249 | 0.99787 | 0.77x |
| 11201855 (new, run 1) | khv5h | 2048 | 37.21 (36.53–39.55) | 72.344 | 0.99818 | 0.99873 | 0.70x |
| 11201855 (new, run 1) | khv4h | 2048 | 31.21 (29.67–32.26) | 67.969 | 0.99186 | 0.99457 | 0.84x |
| 11201855 (new, run 1) | khv5hf | 2048 | 38.88 (37.43–40.12) | 72.344 | 0.99818 | 0.99873 | 0.67x |
| 11201855 (new, run 1) | khv5hp | 2048 | 34.90 (34.48–36.01) | 72.344 | 0.99818 | 0.99873 | 0.75x |
| 11201855 (new, run 1) | khv4hf | 2048 | 32.76 (32.38–36.01) | 67.969 | 0.99186 | 0.99457 | 0.80x |
| 11201855 (new, run 2) | a8 | 2048 | 20.60 (20.12–22.07) | 85.000 | 0.99429 | 0.99909 | 1.27x |
| 11201855 (new, run 2) | k8v4 | 2048 | 26.13 (24.75–31.70) | 69.688 | 0.98678 | 0.99373 | 1.00x |
| 11201855 (new, run 2) | k8v5h | 2048 | 30.70 (30.31–31.33) | 70.469 | 0.99249 | 0.99787 | 0.85x |
| 11201855 (new, run 2) | khv5h | 2048 | 35.90 (35.23–37.24) | 72.344 | 0.99818 | 0.99873 | 0.73x |
| 11201855 (new, run 2) | khv4h | 2048 | 30.63 (30.20–31.36) | 67.969 | 0.99186 | 0.99457 | 0.85x |
| 11201855 (new, run 2) | khv5hf | 2048 | 38.51 (37.79–39.30) | 72.344 | 0.99818 | 0.99873 | 0.68x |
| 11201855 (new, run 2) | khv5hp | 2048 | 36.46 (34.90–39.50) | 72.344 | 0.99818 | 0.99873 | 0.72x |
| 11201855 (new, run 2) | khv4hf | 2048 | 32.45 (32.24–37.85) | 67.969 | 0.99186 | 0.99457 | 0.81x |
| 11201455 (14:55 data) | a8 | 2048 | 20.74 (19.51–21.14) | 85.000 | 0.99699 | 0.99934 | 1.22x |
| 11201455 (14:55 data) | k8v4 | 2048 | 25.21 (24.59–27.62) | 69.688 | 0.99141 | 0.99403 | 1.00x |
| 11201455 (14:55 data) | k8v5h | 2048 | 36.60 (36.46–38.41) | 70.469 | 0.99596 | 0.99818 | 0.69x |
| 11201455 (14:55 data) | khv5h | 2048 | 41.65 (41.16–46.15) | 72.344 | 0.99822 | 0.99878 | 0.61x |
| 11201455 (14:55 data) | khv4h | 2048 | 35.93 (35.19–37.34) | 67.969 | 0.99218 | 0.99462 | 0.70x |
| 11201455 (14:55 data) | khv5hf | 2048 | 44.01 (42.53–46.90) | 72.344 | 0.99822 | 0.99878 | 0.57x |
| 11201455 (14:55 data) | khv5hp | 2048 | 36.67 (35.04–37.53) | 72.344 | 0.99822 | 0.99878 | 0.69x |
| 11201455 (14:55 data) | khv4hf | 2048 | 31.79 (31.42–33.01) | 67.969 | 0.99218 | 0.99462 | 0.79x |
| 11201055 (10:55 data) | a8 | 2048 | 34.14 (32.58–36.18) | 85.000 | 0.99044 | 0.99901 | 1.12x |
| 11201055 (10:55 data) | k8v4 | 2048 | 38.07 (36.65–39.87) | 69.688 | 0.98489 | 0.99352 | 1.00x |
| 11201055 (10:55 data) | k8v5h | 2048 | 50.44 (49.46–52.77) | 70.469 | 0.98845 | 0.99777 | 0.75x |
| 11201055 (10:55 data) | khv5h | 2048 | 59.19 (56.89–65.80) | 72.344 | 0.99791 | 0.99870 | 0.64x |
| 11201055 (10:55 data) | khv4h | 2048 | 40.26 (39.11–53.46) | 67.969 | 0.98940 | 0.99441 | 0.95x |
| 11201055 (10:55 data) | khv5hf | 2048 | 55.35 (54.93–63.99) | 72.344 | 0.99791 | 0.99870 | 0.69x |
| 11201055 (10:55 data) | khv5hp | 2048 | 46.27 (45.48–48.79) | 72.344 | 0.99791 | 0.99869 | 0.82x |
| 11201055 (10:55 data) | khv4hf | 2048 | 43.65 (38.89–44.75) | 67.969 | 0.98940 | 0.99441 | 0.87x |
| 11200555 (05:55 data) | a8 | 2048 | 19.96 (19.40–20.81) | 85.000 | 0.99756 | 0.99910 | 1.29x |
| 11200555 (05:55 data) | k8v4 | 2048 | 25.76 (24.91–26.26) | 69.688 | 0.99138 | 0.99361 | 1.00x |
| 11200555 (05:55 data) | k8v5h | 2048 | 31.70 (30.99–32.28) | 70.469 | 0.99646 | 0.99794 | 0.81x |
| 11200555 (05:55 data) | khv5h | 2048 | 35.35 (34.58–35.98) | 72.344 | 0.99837 | 0.99877 | 0.73x |
| 11200555 (05:55 data) | khv4h | 2048 | 34.83 (29.56–37.30) | 67.969 | 0.99178 | 0.99443 | 0.74x |
| 11200555 (05:55 data) | khv5hf | 2048 | 42.16 (41.13–43.42) | 72.344 | 0.99837 | 0.99877 | 0.61x |
| 11200555 (05:55 data) | khv5hp | 2048 | 41.23 (39.72–45.39) | 72.344 | 0.99836 | 0.99877 | 0.62x |
| 11200555 (05:55 data) | khv4hf | 2048 | 37.11 (35.85–38.09) | 67.969 | 0.99178 | 0.99443 | 0.69x |

### ctx2048 / B8 (best TK per format; attention ms median (min–max) of 7 rounds × 3 reps; KV MiB; min / mean cos vs f32 exact softmax)

| seed | fmt | TK | attn ms med (min–max) | KV MiB | min cos | mean cos | speed vs K8V4 |
|---|---|---|---|---|---|---|---|
| 11201855 (new, run 1) | a8 | 2048 | 6.79 (6.29–7.23) | 21.250 | 0.99745 | 0.99938 | 1.37x |
| 11201855 (new, run 1) | k8v4 | 2048 | 9.29 (9.02–10.19) | 19.062 | 0.99271 | 0.99656 | 1.00x |
| 11201855 (new, run 1) | k8v5h | 2048 | 10.92 (10.71–11.12) | 18.906 | 0.99643 | 0.99874 | 0.85x |
| 11201855 (new, run 1) | khv5h | 2048 | 12.58 (12.28–12.80) | 19.375 | 0.99850 | 0.99932 | 0.74x |
| 11201855 (new, run 1) | khv4h | 1024 | 11.34 (10.85–11.64) | 18.750 | 0.99302 | 0.99715 | 0.82x |
| 11201855 (new, run 1) | khv5hf | 2048 | 13.39 (13.24–13.82) | 19.375 | 0.99850 | 0.99932 | 0.69x |
| 11201855 (new, run 1) | khv5hp | 1024 | 12.61 (12.40–13.01) | 19.375 | 0.99850 | 0.99932 | 0.74x |
| 11201855 (new, run 1) | khv4hf | 2048 | 11.77 (11.65–12.14) | 18.750 | 0.99302 | 0.99715 | 0.79x |
| 11201855 (new, run 2) | a8 | 2048 | 6.54 (6.16–6.97) | 21.250 | 0.99745 | 0.99938 | 1.39x |
| 11201855 (new, run 2) | k8v4 | 1024 | 9.10 (8.97–10.21) | 19.062 | 0.99271 | 0.99656 | 1.00x |
| 11201855 (new, run 2) | k8v5h | 1024 | 11.19 (10.76–11.94) | 18.906 | 0.99643 | 0.99874 | 0.81x |
| 11201855 (new, run 2) | khv5h | 2048 | 12.38 (11.87–13.49) | 19.375 | 0.99850 | 0.99932 | 0.73x |
| 11201855 (new, run 2) | khv4h | 2048 | 11.24 (10.84–11.43) | 18.750 | 0.99302 | 0.99715 | 0.81x |
| 11201855 (new, run 2) | khv5hf | 2048 | 13.07 (12.60–13.39) | 19.375 | 0.99850 | 0.99932 | 0.70x |
| 11201855 (new, run 2) | khv5hp | 2048 | 12.02 (11.84–12.53) | 19.375 | 0.99850 | 0.99932 | 0.76x |
| 11201855 (new, run 2) | khv4hf | 1024 | 11.54 (11.27–12.77) | 18.750 | 0.99302 | 0.99715 | 0.79x |
| 11201455 (14:55 data) | a8 | 2048 | 6.34 (6.14–6.95) | 21.250 | 0.99739 | 0.99938 | 1.47x |
| 11201455 (14:55 data) | k8v4 | 1024 | 9.31 (9.11–9.73) | 19.062 | 0.99402 | 0.99652 | 1.00x |
| 11201455 (14:55 data) | k8v5h | 1024 | 10.98 (10.72–11.22) | 18.906 | 0.99700 | 0.99873 | 0.85x |
| 11201455 (14:55 data) | khv5h | 1024 | 12.83 (12.24–13.07) | 19.375 | 0.99855 | 0.99930 | 0.73x |
| 11201455 (14:55 data) | khv4h | 2048 | 11.36 (11.00–11.73) | 18.750 | 0.99423 | 0.99710 | 0.82x |
| 11201455 (14:55 data) | khv5hf | 2048 | 13.33 (13.05–13.80) | 19.375 | 0.99855 | 0.99930 | 0.70x |
| 11201455 (14:55 data) | khv5hp | 1024 | 12.38 (12.13–12.80) | 19.375 | 0.99855 | 0.99930 | 0.75x |
| 11201455 (14:55 data) | khv4hf | 1024 | 11.86 (11.62–12.08) | 18.750 | 0.99423 | 0.99710 | 0.79x |
| 11201055 (10:55 data) | a8 | 2048 | 7.16 (6.70–7.27) | 21.250 | 0.99450 | 0.99918 | 1.54x |
| 11201055 (10:55 data) | k8v4 | 1024 | 11.01 (9.41–13.56) | 19.062 | 0.99243 | 0.99646 | 1.00x |
| 11201055 (10:55 data) | k8v5h | 1024 | 13.18 (12.43–17.48) | 18.906 | 0.99439 | 0.99857 | 0.84x |
| 11201055 (10:55 data) | khv5h | 1024 | 20.01 (18.61–21.51) | 19.375 | 0.99833 | 0.99932 | 0.55x |
| 11201055 (10:55 data) | khv4h | 2048 | 17.96 (17.32–18.56) | 18.750 | 0.99277 | 0.99723 | 0.61x |
| 11201055 (10:55 data) | khv5hf | 2048 | 20.39 (20.13–21.10) | 19.375 | 0.99833 | 0.99932 | 0.54x |
| 11201055 (10:55 data) | khv5hp | 1024 | 19.39 (19.26–20.20) | 19.375 | 0.99833 | 0.99932 | 0.57x |
| 11201055 (10:55 data) | khv4hf | 2048 | 19.10 (17.72–19.81) | 18.750 | 0.99277 | 0.99723 | 0.58x |
| 11200555 (05:55 data) | a8 | 2048 | 7.14 (6.94–8.15) | 21.250 | 0.99668 | 0.99936 | 1.60x |
| 11200555 (05:55 data) | k8v4 | 1024 | 11.45 (11.32–13.66) | 19.062 | 0.99177 | 0.99701 | 1.00x |
| 11200555 (05:55 data) | k8v5h | 1024 | 14.39 (12.85–15.38) | 18.906 | 0.99642 | 0.99885 | 0.80x |
| 11200555 (05:55 data) | khv5h | 2048 | 15.39 (15.05–25.05) | 19.375 | 0.99867 | 0.99943 | 0.74x |
| 11200555 (05:55 data) | khv4h | 2048 | 11.79 (11.59–12.30) | 18.750 | 0.99428 | 0.99758 | 0.97x |
| 11200555 (05:55 data) | khv5hf | 1024 | 15.54 (15.34–15.72) | 19.375 | 0.99867 | 0.99943 | 0.74x |
| 11200555 (05:55 data) | khv5hp | 1024 | 17.79 (17.33–21.34) | 19.375 | 0.99867 | 0.99943 | 0.64x |
| 11200555 (05:55 data) | khv4hf | 2048 | 18.34 (17.54–20.21) | 18.750 | 0.99428 | 0.99758 | 0.62x |

### ctx8192 / B1 (best TK per format; attention ms median (min–max) of 7 rounds × 3 reps; KV MiB; min / mean cos vs f32 exact softmax)

| seed | fmt | TK | attn ms med (min–max) | KV MiB | min cos | mean cos | speed vs K8V4 |
|---|---|---|---|---|---|---|---|
| 11201855 (new, run 1) | a8 | 2048 | 3.73 (3.08–4.12) | 10.625 | 0.99429 | 0.99844 | 1.18x |
| 11201855 (new, run 1) | k8v4 | 2048 | 4.38 (3.83–4.64) | 8.711 | 0.98678 | 0.99278 | 1.00x |
| 11201855 (new, run 1) | k8v5h | 2048 | 4.84 (4.56–5.48) | 8.809 | 0.99249 | 0.99717 | 0.91x |
| 11201855 (new, run 1) | khv5h | 2048 | 5.87 (5.05–6.32) | 9.043 | 0.99829 | 0.99869 | 0.75x |
| 11201855 (new, run 1) | khv4h | 2048 | 4.35 (4.26–4.62) | 8.496 | 0.99269 | 0.99437 | 1.01x |
| 11201855 (new, run 1) | khv5hf | 2048 | 5.39 (5.12–5.66) | 9.043 | 0.99829 | 0.99869 | 0.81x |
| 11201855 (new, run 1) | khv5hp | 2048 | 5.02 (4.92–7.79) | 9.043 | 0.99829 | 0.99869 | 0.87x |
| 11201855 (new, run 1) | khv4hf | 2048 | 4.66 (4.48–4.80) | 8.496 | 0.99269 | 0.99437 | 0.94x |
| 11201855 (new, run 2) | a8 | 2048 | 3.62 (3.55–3.80) | 10.625 | 0.99429 | 0.99844 | 1.12x |
| 11201855 (new, run 2) | k8v4 | 2048 | 4.06 (3.99–4.49) | 8.711 | 0.98678 | 0.99278 | 1.00x |
| 11201855 (new, run 2) | k8v5h | 2048 | 4.69 (4.52–4.98) | 8.809 | 0.99249 | 0.99717 | 0.87x |
| 11201855 (new, run 2) | khv5h | 2048 | 5.60 (4.93–6.48) | 9.043 | 0.99829 | 0.99869 | 0.73x |
| 11201855 (new, run 2) | khv4h | 2048 | 4.67 (4.39–5.27) | 8.496 | 0.99269 | 0.99437 | 0.87x |
| 11201855 (new, run 2) | khv5hf | 2048 | 5.31 (5.01–5.42) | 9.043 | 0.99829 | 0.99869 | 0.76x |
| 11201855 (new, run 2) | khv5hp | 2048 | 4.88 (4.71–5.31) | 9.043 | 0.99829 | 0.99869 | 0.83x |
| 11201855 (new, run 2) | khv4hf | 2048 | 4.70 (4.58–4.72) | 8.496 | 0.99269 | 0.99437 | 0.87x |
| 11201455 (14:55 data) | a8 | 2048 | 3.21 (3.07–3.55) | 10.625 | 0.99886 | 0.99940 | 1.28x |
| 11201455 (14:55 data) | k8v4 | 2048 | 4.12 (3.91–4.41) | 8.711 | 0.99355 | 0.99432 | 1.00x |
| 11201455 (14:55 data) | k8v5h | 2048 | 4.76 (4.53–4.95) | 8.809 | 0.99782 | 0.99830 | 0.87x |
| 11201455 (14:55 data) | khv5h | 2048 | 5.24 (5.02–5.64) | 9.043 | 0.99864 | 0.99885 | 0.79x |
| 11201455 (14:55 data) | khv4h | 2048 | 4.81 (4.56–5.39) | 8.496 | 0.99434 | 0.99492 | 0.86x |
| 11201455 (14:55 data) | khv5hf | 2048 | 5.24 (5.02–5.75) | 9.043 | 0.99864 | 0.99885 | 0.79x |
| 11201455 (14:55 data) | khv5hp | 2048 | 5.13 (4.92–5.76) | 9.043 | 0.99864 | 0.99885 | 0.80x |
| 11201455 (14:55 data) | khv4hf | 2048 | 4.74 (4.56–4.98) | 8.496 | 0.99434 | 0.99492 | 0.87x |
| 11201055 (10:55 data) | a8 | 2048 | 4.49 (4.21–4.96) | 10.625 | 0.99044 | 0.99721 | 1.13x |
| 11201055 (10:55 data) | k8v4 | 2048 | 5.09 (4.78–5.62) | 8.711 | 0.98489 | 0.99231 | 1.00x |
| 11201055 (10:55 data) | k8v5h | 2048 | 6.82 (6.16–7.65) | 8.809 | 0.98845 | 0.99577 | 0.75x |
| 11201055 (10:55 data) | khv5h | 2048 | 7.10 (6.33–7.77) | 9.043 | 0.99824 | 0.99868 | 0.72x |
| 11201055 (10:55 data) | khv4h | 2048 | 6.27 (5.64–6.79) | 8.496 | 0.99417 | 0.99495 | 0.81x |
| 11201055 (10:55 data) | khv5hf | 2048 | 7.78 (7.17–8.26) | 9.043 | 0.99824 | 0.99868 | 0.65x |
| 11201055 (10:55 data) | khv5hp | 2048 | 7.05 (6.63–8.23) | 9.043 | 0.99824 | 0.99867 | 0.72x |
| 11201055 (10:55 data) | khv4hf | 2048 | 6.42 (6.10–8.10) | 8.496 | 0.99417 | 0.99495 | 0.79x |
| 11200555 (05:55 data) | a8 | 2048 | 3.57 (3.26–4.06) | 10.625 | 0.99776 | 0.99863 | 1.10x |
| 11200555 (05:55 data) | k8v4 | 2048 | 3.94 (3.68–4.00) | 8.711 | 0.99138 | 0.99269 | 1.00x |
| 11200555 (05:55 data) | k8v5h | 2048 | 5.16 (4.63–5.91) | 8.809 | 0.99646 | 0.99735 | 0.76x |
| 11200555 (05:55 data) | khv5h | 2048 | 5.30 (5.09–5.95) | 9.043 | 0.99851 | 0.99867 | 0.74x |
| 11200555 (05:55 data) | khv4h | 2048 | 4.40 (4.26–6.02) | 8.496 | 0.99319 | 0.99399 | 0.90x |
| 11200555 (05:55 data) | khv5hf | 2048 | 5.28 (5.17–5.54) | 9.043 | 0.99851 | 0.99867 | 0.75x |
| 11200555 (05:55 data) | khv5hp | 2048 | 5.06 (4.79–5.29) | 9.043 | 0.99851 | 0.99867 | 0.78x |
| 11200555 (05:55 data) | khv4hf | 2048 | 4.67 (4.42–5.08) | 8.496 | 0.99319 | 0.99399 | 0.84x |

### Totals at B = 8 / ctx 8192 (u32-LUT 2-bit weights 16.65 MiB + KV; int8 weights 66.34 MiB)

| fmt | total MiB | vs int8 + A8 (151.34) | / u32 + K8V4 |
|---|---|---|---|
| a8 | 101.65 | 1.489x | 1.1774x |
| k8v4 | 86.34 | 1.753x | 1.0000x |
| k8v5h | 87.12 | 1.737x | 1.0090x |
| khv5h | 88.99 | 1.701x | 1.0308x |
| khv4h | 84.62 | 1.788x | 0.9801x |
| khv5hf | 88.99 | 1.701x | 1.0308x |
| khv5hp | 88.99 | 1.701x | 1.0308x |
| khv4hf | 84.62 | 1.788x | 0.9801x |

### Prefill-256 hot index: overlap with full-history index, and K rel err (median over streams × heads)

| seed | overlap ctx2048 | overlap ctx8192 | K err khot ctx8192 | K err khotp ctx8192 | K err k8g64 ctx8192 |
|---|---|---|---|---|---|
| 11201855 (new, run 1) | 0.975 | 0.979 | 0.00513 | 0.00514 | 0.01204 |
| 11201455 (14:55 data) | 0.967 | 0.988 | 0.00507 | 0.00507 | 0.01225 |
| 11201055 (10:55 data) | 0.988 | 0.975 | 0.00518 | 0.00518 | 0.01218 |
| 11200555 (05:55 data) | 0.958 | 0.979 | 0.00490 | 0.00491 | 0.01258 |


## Verdict

**viable (narrow).** khv4h is 0.980x K8V4 total bytes (84.62 vs 86.34 MiB, 1.788x vs int8 + A8) and beats K8V4's min cos on all four seeds (+0.0004 to +0.0051). It is not a new headline gain: 2% fewer bytes, at 0.70–0.95x of K8V4 attention speed. It is also 0.006–0.009 min cos below khv5h, which still buys the best accuracy for +5.2% bytes over khv4h.
- Folded 134-wide hot GEMM: **not worth it.** The extra copy into the tile costs more than the second small GEMM saves (khv5hf 0.84–1.07x of khv5h, 0.75–0.91x of k8v5h; khv4hf 0.94–0.95x of khv4h on the new seed).
- Prefill-256 hot index: **free.** Overlap 0.958–0.988, K rel err 0.00491–0.00518 (prefill) vs 0.00490–0.00518 (full history), min cos within 0.00001. A serving path can fix the hot channels at prefill and never re-pick them.
- Recommendation for the i2sd / bitnetd decode cache: khv4h (index from prefill) as the default small cache, khv5h where accuracy matters more than 5% bytes, K8V4 only where attention speed is the constraint.

## Links

- Previous note: [[2026-11-20_1455_f16-scales-k8-a5-g32-old-v-and-hot-k-f16-a8-g32-k-floor-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v5-k8v4-and-all-a8-u32-lut-total-bytes]]
- Related: [[2026-11-20_1055_a5-g32-a4-g16-and-3-tier-a8-a6-a4-old-v-with-k8-k-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-and-all-a8-u32-lut-total-bytes]] (A5 V), [[2026-11-19_1855_top-k-hot-k-channels-a8-plus-per-channel-a4-rest-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-kivi-k4v4-k8v4-and-all-a8-u32-lut-total-bytes]] (first hot-K split)
- This wave's pair files: none. The quant builders are idle.
  - The Quant Chief 18:05 assignment was pair_A no gap and pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-21T01:10:00Z).
  - Chief note: "18:05 11/20: blobs unchanged; no eligible item; both no gap." LLC skip path: `JuniorHome/scripts/advance_gguf_operator_sets_named_20261120.py` (not touched).
  - Roadmap blobs: `JuniorHome docs/ROADMAP.md` 4efb666, `JuniorLLM docs/BETA_TO_OS.md` bac6923.
- bench dir: `/workspace/bench1855_1120` (kh.py, patch.py, run1.txt, run2.txt, xseed1455.txt, xseed1055.txt, xseed0555.txt, results_*.json, tables.md)

## Script (`kh.py`)

```python
#!/usr/bin/env python3
"""Hot-K f16 with A4 V (khv4h), a folded 134-wide hot-K GEMM (khv5hf), and a prefill-chosen hot index (khv5hp) (18:55 11/20). Derived from fs.py (14:55 11/20).
New variants: khv4h old 210 / win 270; khv5hf = khv5h bytes, hot scores folded into one GEMM; khv5hp = khv5h with hot idx from first 256 tokens.
Last WIN=1024 tokens stay A8 g64 for K and V (scale dtype as per variant). Variants (B per token per KV head):
  a8      A8 g64 K/V, f32 scales                            old 272 / win 272   (baseline, all-A8)
  k8v4    A8 g64 K + A4 g32 V old, f32 scales               old 216 / win 272   (current rec, K8V4)
  k8v5    A8 g64 K + A5 g32 V old, f32 scales               old 232 / win 272   (10:55 reference)
  k8v5h   same as k8v5 but every group scale stored f16     old 220 / win 264   (question a)
  a8k32   all-A8, K at g32 (f32), V g64                     old 280 / win 280   (question b, floor)
  a8khot  all-A8, 6 hot K channels/head kept f16, rest A8 g64 old 278 / win 278 (question b, floor)
  khv5h   hot-K f16 + A8 g64 K f16 scales, A5 g32 V old f16 scales, win A8 f16 + hot K: old 226 / win 270
Scales stored f16 are rounded first and the codes computed against the rounded scale.
Hot channels: per (stream, head) top-6 by mean |k| over that stream's history (6 B index per stream-head, counted).
Error-only diagnostics (exact softmax, not timed): kf_v8 (f32 K + A8 V), k8_vf (A8 K + f32 V).
Usage: OPENBLAS_NUM_THREADS=1 SEED=... python3 fs.py out.json
  SEED=11201055 replays 10:55 11/20 data, SEED=11200555 replays 05:55 11/20 data (HR sign draw replayed)."""
import json, sys, time, statistics, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
KVH, G, HD, WIN, W = 5, 4, 128, 1024, 8; SC = np.float32(1 / np.sqrt(HD)); NHOT = 6
CTXS, BS, TKS = (2048, 8192), (1, 8), (1024, 2048)
FMTS = ("a8", "k8v4", "k8v5h", "khv5h", "khv4h", "khv5hf", "khv5hp", "khv4hf")
NPRE = 256
A8FAM = ("a8", "a8k32", "a8khot")
WB = {"u32": 16.65, "int8": 66.34}   # MiB, full bitnetd layer weights from 18:55 11/18 run1 (not re-timed)
ROUNDS, REPS = 7, 3
POOL = ThreadPoolExecutor(W)
SEED = int(os.environ.get("SEED", "11201855")); rng = np.random.default_rng(SEED)
if SEED == 11200555: rng.choice([-1.0, 1.0], HD)   # replay hv4.py's HR sign draw (05:55 11/20 data)
DECU = np.array([[c & 15, c >> 4] for c in range(256)], np.uint8); LUTU16 = DECU.view(np.uint16).ravel()
DEC = np.array([[(c & 15) - 8, (c >> 4) - 8] for c in range(256)], np.int8); LUT16 = DEC.view(np.uint16).ravel()
def scl(g, lim, f16):
    m = np.abs(g).max(-1) / float(lim)
    if f16: m = np.maximum(m, 1e-6).astype(np.float16); return m, m.astype(np.float32)
    m = (m + 1e-12).astype(np.float32); return m, m
def q8(x, gs=64, f16=False):
    g = x.reshape(*x.shape[:-1], HD // gs, gs); ms, m = scl(g, 127.0, f16)
    if not f16: m = np.abs(g).max(-1) / 127.0 + 1e-12   # bit-identical to vw.py
    return np.clip(np.rint(g / m[..., None]), -127, 127).astype(np.int8).reshape(x.shape), ms
def dq8(c, s, gs=64): return (c.reshape(*c.shape[:-1], HD // gs, gs).astype(np.float32) * s.astype(np.float32)[..., None]).reshape(c.shape)
def pack4(c):
    n = c.shape[-1]; c = c.astype(np.uint8).reshape(*c.shape[:-1], n // 2, 2); return (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8)
def qsym(x, lim, gs, f16=False):
    g = x.reshape(*x.shape[:-1], HD // gs, gs); ms, m = scl(g, lim, f16)
    if not f16: m = np.abs(g).max(-1) / float(lim) + 1e-12
    return np.clip(np.rint(g / m[..., None]), -lim, lim).astype(np.int16).reshape(x.shape), (ms if f16 else m.astype(np.float32))
def q4(x, gs=32): c, m = qsym(x, 7, gs); return pack4(c + 8), m
def dq4(p, s, gs=32):
    u = np.take(LUT16, p).view(np.int8).reshape(*p.shape[:-1], HD // gs, gs)
    return (u.astype(np.float32) * s[..., None]).reshape(*p.shape[:-1], HD)
def q5(x, f16=False):
    c, m = qsym(x, 15, 32, f16); c = (c + 16).astype(np.uint8)
    return pack4(c & 15), np.packbits((c >> 4).astype(np.uint8), axis=-1, bitorder="little"), m
def unpack5(lo, hb):
    u = np.take(LUTU16, lo).view(np.uint8).reshape(lo.shape[0], HD)
    u |= np.unpackbits(hb, axis=-1, bitorder="little") << 4; u -= 16; return u.view(np.int8)
def dq5(lo, hb, s): return (unpack5(lo, hb).reshape(lo.shape[0], HD // 32, 32).astype(np.float32) * s.astype(np.float32)[..., None]).reshape(lo.shape[0], HD)
def kvhist(n):  # same generator as 05:55 11/19 - 10:55 11/20
    ch = np.exp(rng.normal(0, 0.5, (KVH, 1, HD))).astype(np.float32); ch[:, :, rng.choice(HD, 6, replace=False)] *= 6
    k = (rng.standard_normal((KVH, n, HD)).astype(np.float32) + 0.3 * np.sin(np.arange(n) / 300.0)[None, :, None]) * ch
    v = rng.standard_normal((KVH, n, HD)).astype(np.float32) * np.exp(rng.normal(0, 0.3, (KVH, 1, HD))).astype(np.float32)
    return k.astype(np.float32), v.astype(np.float32)
def b8(gs, sb): return HD + HD // gs * sb
BOLD = {"a8": 2 * b8(64, 4), "k8v4": b8(64, 4) + 80, "k8v5": b8(64, 4) + 96, "k8v5h": b8(64, 2) + 88,
        "a8k32": b8(32, 4) + b8(64, 4), "a8khot": (HD - NHOT) + 2 * NHOT + 8 + b8(64, 4), "khv5h": (HD - NHOT) + 2 * NHOT + 4 + 88, "khv4h": (HD - NHOT) + 2 * NHOT + 4 + 72}
for _f in ("khv5hf", "khv5hp"): BOLD[_f] = BOLD["khv5h"]
BOLD["khv4hf"] = BOLD["khv4h"]
BWIN = {"a8": 2 * b8(64, 4), "k8v4": 2 * b8(64, 4), "k8v5": 2 * b8(64, 4), "k8v5h": 2 * b8(64, 2),
        "a8k32": BOLD["a8k32"], "a8khot": BOLD["a8khot"], "khv5h": (HD - NHOT) + 2 * NHOT + 4 + b8(64, 2)}
for _f in ("khv4h", "khv5hf", "khv5hp", "khv4hf"): BWIN[_f] = BWIN["khv5h"]
HOTIDX = {"a8khot", "khv5h", "khv4h", "khv5hf", "khv5hp", "khv4hf"}
FOLD = {"khv5hf", "khv4hf"}
class KV:
    def __init__(s, k, v):
        s.n = k.shape[1]; s.kf, s.vf = k, v
        (s.k8, s.ks8), (s.v8, s.vs8) = q8(k), q8(v)
        vd = dq8(s.v8, s.vs8); s.v4, s.v4s = q4(vd); s.v5lo, s.v5hb, s.v5s = q5(vd)
        (s.k8h, s.ks8h), (s.v8h, s.vs8h) = q8(k, 64, True), q8(v, 64, True)
        s.v5hlo, s.v5hhb, s.v5hs = q5(dq8(s.v8h, s.vs8h), True)
        s.k32, s.ks32 = q8(k, 32)
        s.hidx = np.argsort(-np.abs(k).mean(1), axis=1)[:, :NHOT]            # (KVH, 6)
        s.khot = np.stack([k[h][:, s.hidx[h]] for h in range(KVH)]).astype(np.float16)
        kz = k.copy()
        for h in range(KVH): kz[h][:, s.hidx[h]] = 0
        s.kz8, s.kzs8 = q8(kz); s.kz8h, s.kzs8h = q8(kz, 64, True)
        c4, s.v4hs = qsym(dq8(s.v8h, s.vs8h), 7, 32, True); s.v4h = pack4(c4 + 8)
        s.hidxp = np.argsort(-np.abs(k[:, :NPRE]).mean(1), axis=1)[:, :NHOT]
        s.khotp = np.stack([k[h][:, s.hidxp[h]] for h in range(KVH)]).astype(np.float16)
        kzp = k.copy()
        for h in range(KVH): kzp[h][:, s.hidxp[h]] = 0
        s.kzp8h, s.kzps8h = q8(kzp, 64, True)
        s.khot32 = s.khot.astype(np.float32)
    def bytes(s, fmt):
        wb = max(0, s.n - WIN); return KVH * (wb * BOLD[fmt] + (s.n - wb) * BWIN[fmt] + (NHOT if fmt in HOTIDX else 0))
    # K -> scores fn, V dequant, per segment (old / win)
    def S(s, fmt, h, a, b, Q, old):
        if fmt == "a8k32": return Q @ dq8(s.k32[h, a:b], s.ks32[h, a:b], 32).T
        if fmt == "a8khot":
            return Q @ dq8(s.kz8[h, a:b], s.kzs8[h, a:b]).T + Q[:, s.hidx[h]] @ s.khot[h, a:b].astype(np.float32).T
        if fmt in FOLD:
            Kc = np.empty((b - a, HD + NHOT), np.float32); Kc[:, :HD] = dq8(s.kz8h[h, a:b], s.kzs8h[h, a:b]); Kc[:, HD:] = s.khot32[h, a:b]
            return np.concatenate([Q, Q[:, s.hidx[h]]], 1) @ Kc.T
        if fmt == "khv5hp":
            return Q @ dq8(s.kzp8h[h, a:b], s.kzps8h[h, a:b]).T + Q[:, s.hidxp[h]] @ s.khotp[h, a:b].astype(np.float32).T
        if fmt in ("khv5h", "khv4h"):
            return Q @ dq8(s.kz8h[h, a:b], s.kzs8h[h, a:b]).T + Q[:, s.hidx[h]] @ s.khot[h, a:b].astype(np.float32).T
        if fmt == "k8v5h": return Q @ dq8(s.k8h[h, a:b], s.ks8h[h, a:b]).T
        return Q @ dq8(s.k8[h, a:b], s.ks8[h, a:b]).T
    def V(s, fmt, h, a, b, old):
        if old and fmt == "k8v4": return dq4(s.v4[h, a:b], s.v4s[h, a:b])
        if old and fmt == "k8v5": return dq5(s.v5lo[h, a:b], s.v5hb[h, a:b], s.v5s[h, a:b])
        if old and fmt in ("khv4h", "khv4hf"): return dq4(s.v4h[h, a:b], s.v4hs[h, a:b].astype(np.float32))
        if old and fmt in ("k8v5h", "khv5h", "khv5hf", "khv5hp"): return dq5(s.v5hlo[h, a:b], s.v5hhb[h, a:b], s.v5hs[h, a:b])
        if fmt in ("k8v5h", "khv5h", "khv4h", "khv5hf", "khv5hp", "khv4hf"): return dq8(s.v8h[h, a:b], s.vs8h[h, a:b])
        return dq8(s.v8[h, a:b], s.vs8[h, a:b])
def online(m, l, acc, S, Vfn):
    mn = np.maximum(m, S.max(1, keepdims=True)); al = np.exp(m - mn); S -= mn; np.exp(S, out=S)
    l *= al; l += S.sum(1, keepdims=True); acc *= al; acc += Vfn(S); return mn
def tile(kv, fmt, h, a, b, Q, st, old):
    V = kv.V(fmt, h, a, b, old); return online(*st, kv.S(fmt, h, a, b, Q, old), lambda P: P @ V)
def head(kv, Q, h, fmt, TK):
    st = [np.full((G, 1), -np.inf, np.float32), np.zeros((G, 1), np.float32), np.zeros((G, HD), np.float32)]
    if fmt in A8FAM:
        for a in range(0, kv.n, TK): st[0] = tile(kv, fmt, h, a, min(a + TK, kv.n), Q, st, True)
        return st[2] / st[1]
    wb = max(0, kv.n - WIN)
    for a in range(0, wb, TK): st[0] = tile(kv, fmt, h, a, min(a + TK, wb), Q, st, True)
    for a in range(wb, kv.n, WIN): st[0] = tile(kv, fmt, h, a, min(a + WIN, kv.n), Q, st, False)
    return st[2] / st[1]
def attn(kvs, q, fmt, TK):
    futs = [POOL.submit(head, kvs[s], q[s, h], h, fmt, TK) for s in range(len(kvs)) for h in range(KVH)]
    return np.stack([f.result() for f in futs])
def sm(Q, K, V):
    S = Q @ K.T; S -= S.max(1, keepdims=True); P = np.exp(S); P /= P.sum(1, keepdims=True); return P @ V
def ref(kv, Q, h): return sm(Q, kv.kf[h], kv.vf[h])
def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
def main():
    out = {"seed": SEED, "rows": [], "compare": {}, "diag": {}}
    print(f"seed {SEED}\n{'cfg':30s} {'attn ms med (min-max)':>24s} {'KV MiB':>8s} {'min cos':>10s} {'mean cos':>10s}")
    for n in CTXS:
        kvs8 = [KV(*kvhist(n)) for _ in range(max(BS))]
        q8all = (rng.standard_normal((max(BS), KVH, G, HD)).astype(np.float32) * SC)
        pre_hit = np.mean([len(set(kv.hidx[h]) & set(kv.hidxp[h])) / 6 for kv in kvs8 for h in range(KVH)]); out["diag"][f"prefill_idx_overlap_ctx{n}"] = float(pre_hit); print(f"  prefill-256 hot idx overlap with full-history idx ctx{n}: {pre_hit:.3f}")
        planted_hit = np.mean([len(set(kv.hidx[h]) & set(np.argsort(-np.abs(kv.kf[h]).max(0))[:6])) / 6 for kv in kvs8 for h in range(KVH)])
        ke = {nm: float(np.median([np.linalg.norm(f(kv, h) - kv.kf[h]) / np.linalg.norm(kv.kf[h]) for kv in kvs8 for h in range(KVH)])) for nm, f in (
            ("k8g64", lambda kv, h: dq8(kv.k8[h], kv.ks8[h])), ("k8g64h", lambda kv, h: dq8(kv.k8h[h], kv.ks8h[h])),
            ("k8g32", lambda kv, h: dq8(kv.k32[h], kv.ks32[h], 32)),
            ("khotp", lambda kv, h: (lambda K: (K.__setitem__((slice(None), kv.hidxp[h]), kv.khotp[h].astype(np.float32)), K)[1])(dq8(kv.kzp8h[h], kv.kzps8h[h]))),
            ("khot", lambda kv, h: (lambda K: (K.__setitem__((slice(None), kv.hidx[h]), kv.khot[h].astype(np.float32)), K)[1])(dq8(kv.kz8[h], kv.kzs8[h]))))}
        out["diag"][f"Kerr_ctx{n}"] = ke; print(f"  K rel err ctx{n}: " + " ".join(f"{k} {v:.5f}" for k, v in ke.items()) + f" | hot-idx overlap with top-6 absmax {planted_hit:.2f}")
        for B in BS:
            kvs, q = kvs8[:B], q8all[:B]
            R = np.stack([np.stack([ref(kvs[s], q[s, h], h) for h in range(KVH)]) for s in range(B)]).reshape(B * KVH, G, HD)
            for dn, kf_, vf_ in (("kf_v8", lambda kv, h: kv.kf[h], lambda kv, h: dq8(kv.v8[h], kv.vs8[h])),
                                 ("k8_vf", lambda kv, h: dq8(kv.k8[h], kv.ks8[h]), lambda kv, h: kv.vf[h])):
                D = np.stack([np.stack([sm(q[s, h], kf_(kvs[s], h), vf_(kvs[s], h)) for h in range(KVH)]) for s in range(B)]).reshape(B * KVH, G, HD)
                cs = [cos(D[i], R[i]) for i in range(B * KVH)]; out["diag"][f"{dn}_ctx{n}_B{B}"] = dict(min=min(cs), mean=float(np.mean(cs)))
                print(f"  diag {dn} ctx{n} B{B}: min cos {min(cs):.5f} mean {np.mean(cs):.5f}")
            res = {}
            for fmt in FMTS:
                for TK in TKS:
                    o = attn(kvs, q, fmt, TK)
                    for _ in range(2): attn(kvs, q, fmt, TK)
                    ts = []
                    for _ in range(ROUNDS):
                        t0 = time.perf_counter()
                        for _ in range(REPS): attn(kvs, q, fmt, TK)
                        ts.append((time.perf_counter() - t0) / REPS * 1e3)
                    cs = [cos(o[i], R[i]) for i in range(B * KVH)]
                    key = f"{fmt}_TK{TK}_ctx{n}_B{B}"
                    r = dict(key=key, med=statistics.median(ts), lo=min(ts), hi=max(ts), ts=ts, mib=B * kvs[0].bytes(fmt) / 2**20, cos=min(cs), mcos=float(np.mean(cs)),
                             bpt=kvs[0].bytes(fmt) / n)
                    res[(fmt, TK)] = r; out["rows"].append(r)
                    print(f"{key:30s} {r['med']:8.2f} ({r['lo']:.2f}-{r['hi']:.2f})      {r['mib']:8.3f} {r['cos']:10.5f} {r['mcos']:10.5f}")
            best = {f: min((res[(f, t)] for t in TKS), key=lambda r: r["med"]) for f in FMTS}
            for f in FMTS[1:]:
                for refn in ("a8", "k8v4", "k8v5h", "khv5h"):
                    if f == refn: continue
                    k = f"{f}_vs_{refn}_ctx{n}_B{B}"; r = best[refn]["med"] / best[f]["med"]
                    out["compare"][k] = dict(speed=r, bytes=best[refn]["mib"] / best[f]["mib"], dcos=best[f]["cos"] - best[refn]["cos"])
                    print(f"  {k:28s} speed {r:5.2f}x  KV bytes {best[refn]['mib'] / best[f]['mib']:6.4f}x  dcos {best[f]['cos'] - best[refn]['cos']:+.5f}")
            if B == 8:
                for wf in ("u32", "int8"):
                    for f in FMTS: out["compare"][f"total_{wf}_{f}_ctx{n}_B8"] = WB[wf] + best[f]["mib"]
                t = lambda wf, f: out["compare"][f"total_{wf}_{f}_ctx{n}_B8"]
                for f in FMTS:
                    print(f"  total ctx{n} B8 u32+{f}: {t('u32', f):7.3f} MiB | vs int8+a8 {t('int8','a8')/t('u32', f):5.3f}x | vs int8+same {t('int8', f)/t('u32', f):5.3f}x | / u32+k8v4 {t('u32', f)/t('u32','k8v4'):6.4f}x")
    json.dump(out, open(sys.argv[1], "w"), indent=1)
main()
```

## Next question

khv4h is the smallest accurate cache so far, but every kh* format is 0.6–0.95x of K8V4 speed in numpy, and most of that is the f32 dequant of K and V per tile. At B = 8 / ctx 8192 (also ctx 2048, B = 1), on seeds 11201855 / 11201455 / 11201055 / 11200555:
- Does computing scores on int8 K codes directly (`(Q / s) · codes` per 64-channel group, with Q pre-scaled per group and one int8→f32 cast) bring khv4h to ≥ 0.95x of K8V4 attention speed at the same min cos?
- Does applying V's f16 group scales to P (P · s per 32-token group) instead of dequantizing V shrink khv4h's V-side cost?
- With 4 instead of 6 hot channels (khv4h-4, 206 B per old token), how much of the +0.0004 to +0.0051 min cos over K8V4 is kept?

close: sha256(prev || this)[:16] = 3d0cfe5e6cc28724
