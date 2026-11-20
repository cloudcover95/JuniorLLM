---
local: true
date: 2026-11-20
wave: "2026-11-20 14:55 (ran ~14:48–15:00 MT; Chief last_run 2026-11-20T21:10:00Z = 14:10 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-stream, kv-cache, a5-v, f16-scales, hot-k-channels, a8-g32-k, k8v4, k8v5, a8-g64, recent-window, gqa, numpy, viable]
---

# f16 group scales pay for A5 V. **K8 + A5 g32 V with every group scale stored in f16 (k8v5h)** uses **87.12 MiB** total at B = 8 / ctx 8192. That is **1.009x** K8V4's 86.34 MiB, inside the ~1.01x bar, and 1.737x below int8 + A8 (151.34 MiB) vs K8V4's 1.753x. Its min attention cos is within **0.0002** of f32-scale k8v5 in all 12 configs (seed / ctx / B): **0.98845** vs 0.98832 on the 10:55 data, **0.99646** vs 0.99642 on the 05:55 data, 0.99596 vs 0.99589 on this wave's seed. Against K8V4 that is **+0.0036 / +0.0051 / +0.0046** min cos, closing 64% / 82% / 82% of the K8V4 → all-A8 gap for +0.9% bytes. At ctx 2048 it is even smaller than K8V4 (0.996x). Attention runs at 0.92–1.01x of k8v5 and **0.73–0.87x of K8V4** at B = 8 / ctx 8192. The all-A8 floor at seed 11201055 (0.99044) **comes entirely from K quantization**: f32 K + A8 V gives 0.99995. Keeping the **6 hot K channels per head in f16** lifts all-A8 to **0.99984** (1.022x a8 bytes, 0.66–0.88x a8 speed). **A8 g32 K** gets only to 0.99514 (1.029x bytes, ~1.03x speed, 0.78x on one run). Hot-K f16 + K8 f16 + A5 V f16 (khv5h) reaches **0.99791–0.99837** at B = 8 / ctx 8192 on all three seeds, better than all-A8 on every seed, at 88.99 MiB (1.031x K8V4, 1.70x vs int8 + A8). It costs 0.53–0.80x of K8V4's speed in this numpy loop. Verdict: **viable** in a narrow sense: memory stays ≥ 1.5x vs int8 + A8 (1.74x) with a real accuracy gain over K8V4 at ~equal bytes. It is not a new memory or speed record. Recommend k8v5h over K8V4 where ~0.8x attention speed is acceptable.

## Question

[[2026-11-20_1055_a5-g32-a4-g16-and-3-tier-a8-a6-a4-old-v-with-k8-k-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-and-all-a8-u32-lut-total-bytes]] found A5 g32 old V closes 62% / 82% of the K8V4 → all-A8 gap but costs 1.051x bytes. Its next question:
- At B = 8 / ctx 8192 (also ctx 2048 and B = 1), does K8 + A5 g32 old V with **f16 instead of f32 group scales** (on old K, old V and the A8 1024 window) land within ~1.01x of K8V4 total bytes, with min attention cos within 0.0005 of f32-scale k8v5 (0.98832 on seed 11201055 / 0.99642 on the 05:55 data)?
- At seed 11201055, where all-A8 stops at 0.99044, does **A8 g32 K** or **the 6 hot K channels kept in f16** lift that floor, and at what attention speed?

## Method

- `fs.py` is derived from `vw.py` (10:55 11/20). It uses the same attention-only microbench and the same synthetic K/V generator: Gaussian K/V with a log-normal per-channel spread (K σ 0.5 plus 6 planted 6x hot channels per head, V σ 0.3) and slow positional drift on K. The RNG stream is consumed in the same order.
- Seeds:
  - **11201455** is new and was run twice for timing. Error was identical in both runs.
  - **11201055** replays the 10:55 data. a8 / k8v4 / k8v5 reproduce 0.99044 / 0.98489 / 0.98832 exactly.
  - **11200555** replays the 05:55 data, including hv4.py's HR sign draw. a8 / k8v4 / k8v5 reproduce 0.99756 / 0.99138 / 0.99642 exactly.
- Shapes: B ∈ {1, 8} streams × 5 KV heads × 4 q heads (GQA 20/5), HD 128, ctx ∈ {2048, 8192}. The last 1024 tokens are the A8 g64 window.
- Variants (bytes per token per KV head, old / window):
  - `a8`: A8 g64 K and V, f32 scales. 272 / 272. Baseline all-A8.
  - `k8v4`: A8 g64 K + A4 g32 old V, f32. 216 / 272. Current rec.
  - `k8v5`: A8 g64 K + A5 g32 old V, f32. 232 / 272. 10:55 ref.
  - `k8v5h`: k8v5 with every group scale stored f16 (old K, old V, window K/V). 220 / 264. Each scale is rounded to f16 first, and the codes are computed against the rounded scale.
  - `a8k32`: all-A8, K at g32 (f32 scales), V g64. 280 / 280.
  - `a8khot`: all-A8. The 6 hot K channels per (stream, head) are stored f16 and zeroed before A8 g64 quant of the rest (122 codes counted). 278 / 278, plus a 6 B index per stream-head. Hot channels are the top 6 by mean |k| over the stream's history, with 0.95–0.97 overlap with the top-6 abs-max channels. Scores are computed as `Q·dqK^T + Q[:, hot]·Khot^T`.
  - `khv5h`: hot-K f16 + A8 g64 K f16 scales + A5 g32 old V f16 scales. Window: hot-K + A8 f16. 226 / 270.
- Attention: online softmax per (stream, KV head) on a W = 8 thread pool, `OPENBLAS_NUM_THREADS=1`, TK ∈ {1024, 2048}, best TK per variant, 7 rounds × 3 reps, median (min–max).
- Error: min and mean over (stream, head) of cos(attention output, exact f32-KV softmax). Error-only diagnostics, using exact softmax and not timed: f32 K + A8 g64 V, and A8 g64 K + f32 V.
- Total bytes = full bitnetd layer weights from 18:55 11/18 run1 (u32 LUT 16.65 MiB, int8 66.34 MiB; not re-timed) + KV measured at B = 8.

## Command

```
cd /workspace/bench1455_1120
OPENBLAS_NUM_THREADS=1 SEED=11201455 python3 fs.py results_run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 SEED=11201455 python3 fs.py results_run2.json > run2.txt
OPENBLAS_NUM_THREADS=1 SEED=11201055 python3 fs.py results_xseed1055.json > xseed1055.txt   # 10:55 data replay
OPENBLAS_NUM_THREADS=1 SEED=11200555 python3 fs.py results_xseed0555.json > xseed0555.txt   # 05:55 data replay
```

## Results: attention ms per decode step, all streams (best TK median; run1 / run2 at seed 11201455 | replay 11201055 | replay 11200555)

| ctx / B | a8 | k8v4 | k8v5 | k8v5h | a8k32 | a8khot | khv5h |
|---|---|---|---|---|---|---|---|
| 2048 / 1 | 1.10 / 1.03 \| 0.98 \| 1.18 | 1.56 / 1.45 \| 1.29 \| 1.19 | 1.92 / 1.58 \| 1.59 \| 1.32 | 1.94 / 1.53 \| 1.46 \| 1.46 | 1.02 / 1.13 \| 0.96 \| 0.93 | 1.28 / 1.10 \| 1.04 \| 1.04 | 2.00 / 1.88 \| 1.61 \| 1.83 |
| 2048 / 8 | 6.83 / 6.37 \| 6.05 \| 6.30 | 10.79 / 10.09 \| 9.67 \| 9.53 | 12.83 / 10.81 \| 10.59 \| 10.44 | 14.07 / 11.61 \| 11.28 \| 11.07 | 8.22 / 6.11 \| 6.17 \| 6.23 | 9.23 / 6.97 \| 7.25 \| 6.91 | 17.22 / 13.66 \| 12.61 \| 12.51 |
| 8192 / 1 | 3.79 / 3.19 \| 3.40 \| 3.32 | 3.82 / 3.95 \| 3.76 \| 4.07 | 4.71 / 4.86 \| 4.65 \| 4.40 | 5.22 / 6.34 \| 4.58 \| 4.63 | 3.44 / 4.11 \| 3.53 \| 3.20 | 3.77 / 3.95 \| 3.49 \| 3.64 | 5.16 / 5.59 \| 5.11 \| 5.13 |
| 8192 / 8 | 23.18 / 21.41 \| 20.78 \| 19.95 | 24.81 / 28.09 \| 28.07 \| 25.06 | 32.69 / 32.54 \| 32.29 \| 31.43 | 32.70 / 32.18 \| 34.89 \| 34.18 | 21.95 / 20.53 \| 20.15 \| 25.58 | 30.01 / 24.39 \| 23.64 \| 30.10 | 37.81 / 39.72 \| 35.30 \| 47.32 |

Spread at B = 8 / ctx 8192 (best TK, min–max ms; run1 | run2 | x1055 | x0555):
- a8: 20.24–27.23 | 20.61–23.54 | 19.90–22.38 | 19.74–20.84
- k8v4: 24.67–27.25 | 25.91–32.65 | 26.11–31.07 | 24.54–26.20
- k8v5: 31.39–36.69 | 31.00–32.91 | 31.48–34.17 | 30.94–32.03
- k8v5h: 31.19–34.18 | 31.51–33.64 | 32.36–40.91 | 31.39–35.63
- a8k32: 20.31–23.42 | 19.42–22.12 | 19.65–21.66 | 24.57–26.12
- a8khot: 27.36–30.80 | 22.86–26.23 | 23.45–26.83 | 29.96–31.49
- khv5h: 35.70–43.31 | 36.53–42.10 | 34.55–36.54 | 47.21–48.31

Speed ratios (best-TK median; run1 / run2 | x1055 | x0555):

| ctx / B | k8v5h vs k8v4 | k8v5h vs k8v5 | a8k32 vs a8 | a8khot vs a8 | khv5h vs k8v4 | khv5h vs k8v5 |
|---|---|---|---|---|---|---|
| 2048 / 1 | 0.81 / 0.95 \| 0.89 \| 0.81 | 0.99 / 1.03 \| 1.09 \| 0.90 | 1.07 / 0.92 \| 1.02 \| 1.27 | 0.85 / 0.94 \| 0.94 \| 1.14 | 0.78 / 0.77 \| 0.80 \| 0.65 | 0.96 / 0.84 \| 0.99 \| 0.72 |
| 2048 / 8 | 0.77 / 0.87 \| 0.86 \| 0.86 | 0.91 / 0.93 \| 0.94 \| 0.94 | 0.83 / 1.04 \| 0.98 \| 1.01 | 0.74 / 0.91 \| 0.84 \| 0.91 | 0.63 / 0.74 \| 0.77 \| 0.76 | 0.75 / 0.79 \| 0.84 \| 0.83 |
| 8192 / 1 | 0.73 / 0.62 \| 0.82 \| 0.88 | 0.90 / 0.77 \| 1.02 \| 0.95 | 1.10 / 0.78 \| 0.96 \| 1.04 | 1.01 / 0.81 \| 0.97 \| 0.91 | 0.74 / 0.71 \| 0.74 \| 0.79 | 0.91 / 0.87 \| 0.91 \| 0.86 |
| 8192 / 8 | 0.76 / 0.87 \| 0.80 \| 0.73 | 1.00 / 1.01 \| 0.93 \| 0.92 | 1.06 / 1.04 \| 1.03 \| 0.78 | 0.77 / 0.88 \| 0.88 \| 0.66 | 0.66 / 0.71 \| 0.80 \| 0.53 | 0.86 / 0.82 \| 0.91 \| 0.66 |

## Results: min / mean attention cos vs exact f32 attention (identical in run1 and run2)

Seed 11201455 (this wave):

| ctx / B | a8 | k8v4 | k8v5 | k8v5h | a8k32 | a8khot | khv5h | diag f32 K + A8 V | diag A8 K + f32 V |
|---|---|---|---|---|---|---|---|---|---|
| 2048 / 1 | 0.99863 / 0.99936 | 0.99520 / 0.99631 | 0.99854 / 0.99871 | 0.99855 / 0.99872 | 0.99941 / 0.99960 | 0.99991 / 0.99993 | 0.99894 / 0.99930 | 0.99997 | 0.99865 |
| 2048 / 8 | 0.99739 / 0.99938 | 0.99402 / 0.99652 | 0.99703 / 0.99874 | 0.99700 / 0.99873 | 0.99854 / 0.99962 | 0.99984 / 0.99994 | 0.99855 / 0.99930 | 0.99996 | 0.99744 |
| 8192 / 1 | 0.99886 / 0.99940 | 0.99355 / 0.99432 | 0.99781 / 0.99830 | 0.99782 / 0.99830 | 0.99890 / 0.99950 | 0.99992 / 0.99993 | 0.99864 / 0.99885 | 0.99997 | 0.99889 |
| 8192 / 8 | 0.99699 / 0.99934 | 0.99141 / 0.99403 | 0.99589 / 0.99818 | 0.99596 / 0.99818 | 0.99853 / 0.99964 | 0.99986 / 0.99993 | 0.99822 / 0.99878 | 0.99997 | 0.99709 |

Seed 11201055 replay (10:55 11/20 data; a8 / k8v4 / k8v5 match the 10:55 note exactly):

| ctx / B | a8 | k8v4 | k8v5 | k8v5h | a8k32 | a8khot | khv5h | diag f32 K + A8 V | diag A8 K + f32 V |
|---|---|---|---|---|---|---|---|---|---|
| 2048 / 1 | 0.99876 / 0.99941 | 0.99476 / 0.99633 | 0.99769 / 0.99864 | 0.99762 / 0.99862 | 0.99941 / 0.99964 | 0.99989 / 0.99992 | 0.99906 / 0.99923 | 0.99998 | 0.99880 |
| 2048 / 8 | 0.99450 / 0.99918 | 0.99243 / 0.99646 | 0.99433 / 0.99857 | 0.99439 / 0.99857 | 0.99759 / 0.99958 | 0.99987 / 0.99993 | 0.99833 / 0.99932 | 0.99996 | 0.99452 |
| 8192 / 1 | 0.99044 / 0.99721 | 0.98489 / 0.99231 | 0.98832 / 0.99581 | 0.98845 / 0.99577 | 0.99514 / 0.99856 | 0.99988 / 0.99991 | 0.99824 / 0.99868 | 0.99997 | 0.99043 |
| 8192 / 8 | 0.99044 / 0.99901 | 0.98489 / 0.99352 | 0.98832 / 0.99779 | 0.98845 / 0.99777 | 0.99514 / 0.99943 | 0.99984 / 0.99993 | 0.99791 / 0.99870 | 0.99995 | 0.99043 |

Seed 11200555 replay (05:55 11/20 data; a8 / k8v4 / k8v5 match exactly):

| ctx / B | a8 | k8v4 | k8v5 | k8v5h | a8k32 | a8khot | khv5h | diag f32 K + A8 V | diag A8 K + f32 V |
|---|---|---|---|---|---|---|---|---|---|
| 2048 / 1 | 0.99668 / 0.99869 | 0.99449 / 0.99667 | 0.99643 / 0.99828 | 0.99642 / 0.99826 | 0.99839 / 0.99936 | 0.99990 / 0.99992 | 0.99902 / 0.99949 | 0.99997 | 0.99670 |
| 2048 / 8 | 0.99668 / 0.99936 | 0.99177 / 0.99701 | 0.99643 / 0.99886 | 0.99642 / 0.99885 | 0.99817 / 0.99960 | 0.99988 / 0.99993 | 0.99867 / 0.99943 | 0.99996 | 0.99670 |
| 8192 / 1 | 0.99776 / 0.99863 | 0.99138 / 0.99269 | 0.99666 / 0.99736 | 0.99646 / 0.99735 | 0.99605 / 0.99830 | 0.99986 / 0.99992 | 0.99851 / 0.99867 | 0.99997 | 0.99781 |
| 8192 / 8 | 0.99756 / 0.99910 | 0.99138 / 0.99361 | 0.99642 / 0.99796 | 0.99646 / 0.99794 | 0.99605 / 0.99941 | 0.99986 / 0.99993 | 0.99837 / 0.99877 | 0.99997 | 0.99761 |

k8v5h − k8v5 min cos (f16 vs f32 scales), all 4 configs:
- 11201455: 2048/1 +0.00001, 2048/8 -0.00004, 8192/1 +0.00001, 8192/8 +0.00007
- 11201055: 2048/1 -0.00007, 2048/8 +0.00006, 8192/1 +0.00013, 8192/8 +0.00013
- 11200555: 2048/1 -0.00001, 2048/8 -0.00001, 8192/1 -0.00020, 8192/8 +0.00003

Share of the K8V4 → all-A8 min-cos gap closed at B = 8 / ctx 8192:
- 11201455: k8v5 80%, k8v5h 82%, khv5h 122%
- 11201055: k8v5 62%, k8v5h 64%, khv5h 235%
- 11200555: k8v5 82%, k8v5h 82%, khv5h 113%

Median K relative Frobenius error vs f32 K (seed 11201455, all tokens, ctx 8192): k8g64 0.01225, k8g64h 0.01225, k8g32 0.00954, khot 0.00507

## Results: bytes at B = 8

| variant | B / old token / head | B / window token / head | KV MiB ctx 2048 | KV MiB ctx 8192 | total ctx 8192 (u32 + KV) | vs int8 + A8 (151.34 MiB) | vs int8 + same KV | total / u32 + K8V4 |
|---|---|---|---|---|---|---|---|---|
| a8 | 272 | 272 | 21.250 | 85.000 | 101.650 | 1.489x | 1.489x | 1.177x |
| k8v4 | 216 | 272 | 19.062 | 69.688 | 86.338 | 1.753x | 1.576x | 1.000x |
| k8v5 | 232 | 272 | 19.688 | 74.062 | 90.713 | 1.668x | 1.548x | 1.051x |
| **k8v5h** | 220 | 264 | 18.906 | 70.469 | **87.119** | **1.737x** | 1.570x | **1.009x** |
| a8k32 | 280 | 280 | 21.875 | 87.500 | 104.150 | 1.453x | 1.477x | 1.206x |
| a8khot | 278 | 278 | 21.719 | 86.875 | 103.525 | 1.462x | 1.480x | 1.199x |
| khv5h | 226 | 270 | 19.375 | 72.344 | 88.994 | 1.701x | 1.558x | 1.031x |

At ctx 2048 / B = 8 the totals vs int8 + A8 are a8 2.311x, k8v4 2.453x, k8v5 2.410x, **k8v5h 2.463x (0.996x K8V4 bytes)**, a8k32 2.274x, a8khot 2.283x, khv5h 2.431x. At ctx 2048, 1024 of 2048 tokens are window tokens, and the f16 window saves more than A5 V costs. KV per-stream B/token (5 heads, ctx 8192): k8v4 1115.0, k8v5 1185.0, k8v5h 1127.5, khv5h 1157.5, a8 1360.0.

## Verdict

**viable** (narrow: accuracy at ~equal bytes, not a new memory or speed record).
- **k8v5h passes both pre-set bars.** Total bytes are 1.009x K8V4 (≤ ~1.01x). Min cos is within 0.0002 of f32-scale k8v5 in all 12 configs (worst −0.00020, 05:55 data ctx 8192 / B = 1), against a 0.0005 bar. f16 scales cost nothing measurable in accuracy: the f16 scale rounding error (~5e-4 relative) is far below the A5/A8 step size. Memory vs int8 + A8 stays **1.74x** (≥ 1.5x), and min cos improves over K8V4 by **+0.0036–0.0051** at B = 8 / ctx 8192 on three seeds. Cost: **0.73–0.87x** K8V4 attention speed at B = 8 / ctx 8192 in numpy, the same ~0.8x that k8v5 already paid. The f16 → f32 scale cast is ~free (0.92–1.01x of k8v5).
- **Seed 11201055's all-A8 floor is a K problem, not a V or softmax problem.** f32 K + A8 V gives 0.99995 there, while A8 K + f32 V gives 0.99043. The 6 planted hot K channels force big g64 scales onto the other 58 channels in their group. **Hot-K f16** removes that: a8khot reaches 0.99984–0.99992 on all seeds and configs (K rel error 0.0123 → 0.0051) at 1.022x a8 bytes and 0.66–0.88x a8 speed at B = 8 / ctx 8192. **A8 g32 K** only halves the group damage (0.99514 / 0.99605 / 0.99853) at 1.029x bytes. It runs ~1.03–1.06x a8 speed on 3 of 4 runs, 0.78x on the 05:55 replay run (box noise). Neither all-A8 variant is a memory win (≤ 1.46x vs int8 + A8). They locate the floor.
- **khv5h** (hot-K f16 + A5 V f16) is the most accurate low-memory layout measured so far: min cos 0.99791 / 0.99837 / 0.99822 at B = 8 / ctx 8192, above all-A8 on every seed, at 1.031x K8V4 bytes (1.70x vs int8 + A8). Its numpy speed is poor, at 0.53–0.80x of K8V4, because of the extra 6-wide hot-channel GEMM per tile. That speed cost is the open item.
- Recommendation: **k8v5h replaces K8V4** as the default old-token cache where ~0.8x attention speed is acceptable: +0.9% bytes, 64–82% of the accuracy gap to all-A8 recovered. If accuracy matters more than the last 3% of bytes, use khv5h, once its hot-channel path is fused.

Caveats: synthetic Gaussian K/V with planted hot channels, which favours hot-channel handling. The hot-channel index is chosen from the full history (oracle-ish), and a real cache must pick it from prefill. All timings are numpy, not a fused kernel. Box noise moves single ratios by ±0.1.

## Links

- previous: [[2026-11-20_1055_a5-g32-a4-g16-and-3-tier-a8-a6-a4-old-v-with-k8-k-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-and-all-a8-u32-lut-total-bytes]]
- related: [[2026-11-20_0255_per-channel-a5-a6-k-old-kv-plus-a4-v-and-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-k4v4-and-all-a8-u32-lut-total-bytes]] (A5/A6 on K; packers reused), [[2026-11-19_1855_top-k-hot-k-channels-a8-plus-per-channel-a4-rest-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-kivi-k4v4-k8v4-and-all-a8-u32-lut-total-bytes]] (earlier hot-K-channel split, A8 hot channels; this wave keeps them f16 on an A8 base)
- This wave's pair files: none. The quant builders are idle.
  - The Quant Chief 14:05 assignment was pair_A no gap and pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-20T21:10:00Z).
  - Chief note: "14:05 11/20: blobs unchanged; no eligible item; both no gap." LLC skip path: `JuniorHome/scripts/advance_gguf_operator_sets_named_20261120.py` (not touched).
  - Roadmap blobs: `JuniorHome docs/ROADMAP.md` 4efb666, `JuniorLLM docs/BETA_TO_OS.md` bac6923.
- bench dir: `/workspace/bench1455_1120` (fs.py, run1.txt, run2.txt, xseed1055.txt, xseed0555.txt, results_*.json, tables.md)

## Script (`fs.py`)

```python
#!/usr/bin/env python3
"""f16 group scales on K8 + A5 V, and lifting the all-A8 floor via K (14:55 11/20). Derived from vw.py (10:55 11/20).
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
FMTS = ("a8", "k8v4", "k8v5", "k8v5h", "a8k32", "a8khot", "khv5h")
A8FAM = ("a8", "a8k32", "a8khot")
WB = {"u32": 16.65, "int8": 66.34}   # MiB, full bitnetd layer weights from 18:55 11/18 run1 (not re-timed)
ROUNDS, REPS = 7, 3
POOL = ThreadPoolExecutor(W)
SEED = int(os.environ.get("SEED", "11201455")); rng = np.random.default_rng(SEED)
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
        "a8k32": b8(32, 4) + b8(64, 4), "a8khot": (HD - NHOT) + 2 * NHOT + 8 + b8(64, 4), "khv5h": (HD - NHOT) + 2 * NHOT + 4 + 88}
BWIN = {"a8": 2 * b8(64, 4), "k8v4": 2 * b8(64, 4), "k8v5": 2 * b8(64, 4), "k8v5h": 2 * b8(64, 2),
        "a8k32": BOLD["a8k32"], "a8khot": BOLD["a8khot"], "khv5h": (HD - NHOT) + 2 * NHOT + 4 + b8(64, 2)}
HOTIDX = {"a8khot", "khv5h"}
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
    def bytes(s, fmt):
        wb = max(0, s.n - WIN); return KVH * (wb * BOLD[fmt] + (s.n - wb) * BWIN[fmt] + (NHOT if fmt in HOTIDX else 0))
    # K -> scores fn, V dequant, per segment (old / win)
    def S(s, fmt, h, a, b, Q, old):
        if fmt == "a8k32": return Q @ dq8(s.k32[h, a:b], s.ks32[h, a:b], 32).T
        if fmt == "a8khot":
            return Q @ dq8(s.kz8[h, a:b], s.kzs8[h, a:b]).T + Q[:, s.hidx[h]] @ s.khot[h, a:b].astype(np.float32).T
        if fmt == "khv5h":
            return Q @ dq8(s.kz8h[h, a:b], s.kzs8h[h, a:b]).T + Q[:, s.hidx[h]] @ s.khot[h, a:b].astype(np.float32).T
        if fmt == "k8v5h": return Q @ dq8(s.k8h[h, a:b], s.ks8h[h, a:b]).T
        return Q @ dq8(s.k8[h, a:b], s.ks8[h, a:b]).T
    def V(s, fmt, h, a, b, old):
        if old and fmt == "k8v4": return dq4(s.v4[h, a:b], s.v4s[h, a:b])
        if old and fmt == "k8v5": return dq5(s.v5lo[h, a:b], s.v5hb[h, a:b], s.v5s[h, a:b])
        if old and fmt in ("k8v5h", "khv5h"): return dq5(s.v5hlo[h, a:b], s.v5hhb[h, a:b], s.v5hs[h, a:b])
        if fmt in ("k8v5h", "khv5h"): return dq8(s.v8h[h, a:b], s.vs8h[h, a:b])
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
        planted_hit = np.mean([len(set(kv.hidx[h]) & set(np.argsort(-np.abs(kv.kf[h]).max(0))[:6])) / 6 for kv in kvs8 for h in range(KVH)])
        ke = {nm: float(np.median([np.linalg.norm(f(kv, h) - kv.kf[h]) / np.linalg.norm(kv.kf[h]) for kv in kvs8 for h in range(KVH)])) for nm, f in (
            ("k8g64", lambda kv, h: dq8(kv.k8[h], kv.ks8[h])), ("k8g64h", lambda kv, h: dq8(kv.k8h[h], kv.ks8h[h])),
            ("k8g32", lambda kv, h: dq8(kv.k32[h], kv.ks32[h], 32)),
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
                for refn in ("a8", "k8v4", "k8v5"):
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

Hot-K f16 fixes K, and A5 V with f16 scales fixes V. Next, try to get both at ≤ K8V4 bytes and speed. At B = 8 / ctx 8192 (also ctx 2048, B = 1), on seeds 11201455 / 11201055 / 11200555:
- Does **khv4h** (6 hot K channels f16 + A8 g64 K f16 scales + A4 g32 old V f16 scales, 210 B per old token, 0.98x K8V4 KV) beat K8V4's min cos (0.99141 / 0.98489 / 0.99138) at ≤ 1.0x total bytes?
- Does folding the hot-channel scores into the main GEMM (append Q[:, hot] / Khot as 6 extra f32 columns of a 134-wide tile) bring khv5h to ≥ 0.95x of k8v5h attention speed?
- How much min cos is lost when the hot-channel index is chosen from the first 256 (prefill) tokens instead of the full history?

close: sha256(prev || this)[:16] = e967e1f245d37f6f
