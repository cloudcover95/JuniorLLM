---
local: true
date: 2026-11-07
wave: "2026-11-07 10:55 (ran ~11:01–11:08 MT; Chief last_tips ~10:08 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 6 / JuniorOS overlay: bitnetd on 127.0.0.1:8765; JuniorHome docs/ROADMAP.md #3 i2sd userspace 127.0.0.1:8767 (Winsor act quant at bitnetd boundary)"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, i2sd, winsor, absmean, int8, activation, token-batch, numpy, not-viable]
---

# Winsor-clip before absmean I2_S does not cut activation mae and slows the bitnetd pack/expand path

## Question

Prior note [[2026-11-07_0555_bitnetd-handler-i2s-block-decode-vs-int8-token-batch]] kept I2_S viable on memory (4.00x vs int8) but not on decode speed, and asked whether a Winsor clip before absmean-ternarize of the per-request activation tile reduces mae vs f32 enough to matter without losing the 4x packed-byte win, versus absmax int8 at B=1/8/32. Quant Chief `last_tips.json` `last_run` 2026-11-07T17:08:00Z (~10:08 MT) still wrote pair_A/pair_B **no gap** (roadmap blobs 4efb666/bac6923); no pair item to prefer.

## Method

Use shipped `junior_bitnet.winsor.pack` semantics (tau = pct(|x|), gamma = mean(min(|x|, tau))+1e-7, trit = clip(round(x/gamma))) versus plain absmean and absmax int8. Activations: Student-t(3)×0.05 (heavier tails so Winsor can bite), H=2560, B=1/8/32, seed `202611071055`. Percentiles p90/p95/p99. Pack I2_S 2B4T (same LUT bit order as 05:55). Time **encode + LUT/int8 decode into a float32 handler buffer** (bitnetd boundary), 3 warm + 7 timed repeats, passes 80/40/20 by B. numpy only, offline. Local tree `/workspace/JuniorLLM`. Command (scratch `/workspace/bench1055_1107/`):

```
PYTHONPATH=/workspace/JuniorLLM python3 bench.py
```

## Results

I2_S packed size stays exactly **4.00x** smaller than int8 at every B (Winsor does not change trit width). Winsor **raises** mae vs f32 relative to plain absmean on every percentile (worse, not better). Encode+decode is slower than plain absmean and far behind int8.

| B | N | I2_S B | int8 B | mem | absmean mae | int8 mae | absmean µs med (min–max) | int8 µs med (min–max) |
|---|---|---|---|---|---|---|---|---|
| 1 | 2560 | 640 | 2560 | **4.00x** | 0.02996 | 0.00201 | 32.54 (32.00–77.45) | 11.28 (11.16–12.76) |
| 8 | 20480 | 5120 | 20480 | **4.00x** | 0.02939 | 0.00386 | 301.61 (299.82–314.19) | 83.60 (82.32–88.08) |
| 32 | 81920 | 20480 | 81920 | **4.00x** | 0.02895 | 0.00620 | 1469.22 (1399.30–1605.14) | 468.91 (458.04–475.05) |

Winsor vs absmean (mae % = change vs absmean; negative = worse; speed = absmean_med / winsor_med):

| B | p | winsor mae | mae vs absmean | winsor µs med (min–max) | speed vs absmean | speed vs int8 |
|---|---|---|---|---|---|---|
| 1 | 90 | 0.03070 | -2.49% | 43.05 (41.89–45.04) | 0.76x | 0.26x |
| 1 | 95 | 0.03031 | -1.17% | 43.89 (41.99–45.36) | 0.74x | 0.26x |
| 1 | 99 | 0.03006 | -0.34% | 43.68 (42.04–45.75) | 0.75x | 0.26x |
| 8 | 90 | 0.03030 | -3.10% | 542.32 (538.91–552.85) | 0.56x | 0.15x |
| 8 | 95 | 0.02986 | -1.62% | 468.37 (461.10–485.79) | 0.64x | 0.18x |
| 8 | 99 | 0.02953 | -0.48% | 551.12 (546.52–556.86) | 0.55x | 0.15x |
| 32 | 90 | 0.02984 | -3.06% | 2577.57 (2475.60–2722.46) | 0.57x | 0.18x |
| 32 | 95 | 0.02943 | -1.65% | 2081.68 (2054.64–2183.47) | 0.71x | 0.23x |
| 32 | 99 | 0.02909 | -0.46% | 2536.21 (2490.64–2767.89) | 0.58x | 0.18x |

Numpy winsor path matched `junior_bitnet.winsor.pack` trit codes on a 256-probe; I2_S pack matched `junior_bitnet.i2s.pack`. Exact samples in scratch `results.json` (numpy 2.5.4).

## Verdict

**Not viable.** Winsor-clip-before-absmean on per-request activation tiles does **not** buy mae vs f32 (p90–p99 are 0.3–3.1% *worse* than plain absmean) and adds encode cost (0.55–0.76x of absmean, 0.15–0.26x of absmax int8). The 4.00x packed-byte win is already present with plain absmean I2_S and is unchanged by Winsor. Int8 still owns reconstruction error (~0.002–0.006 vs ~0.030). Keep plain absmean for activation ternarize at the bitnetd boundary; reserve Winsor for the weight-side path it was written for. Bounded to Student-t(3) tiles H=2560 B≤32 on a shared box.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-07T17:08:00Z (~10:08 MT) wrote pair_A/pair_B **no gap**. Roadmap blobs still JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`. JuniorLLM tip before this push: [1a78af3](https://github.com/cloudcover95/JuniorLLM/commit/1a78af3) (prior vault note).

## Next question

Does a **weight-resident** I2_S LUT GEMV (score activations against packed trit weights without expanding weights to float32) recover ≥1.5x speed vs int8-expand+GEMV at B=1/8 while keeping the 4x weight-memory win — i.e. the path 05:55 said to prefer over decode-into-float32?

close: 17db2293ca90fd98 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-07_0555)
