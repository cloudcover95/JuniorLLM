---
local: true
date: 2026-11-07
wave: "2026-11-07 14:55 (ran ~15:03–15:06 MT; Chief last_tips ~14:09 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 6 / JuniorOS overlay: bitnetd on 127.0.0.1:8765; JuniorHome docs/ROADMAP.md #3 i2sd userspace 127.0.0.1:8767 (weight-resident I2_S LUT GEMV)"
verdict: viable
tags: [osai, vault, findings, ternary, bitnetd, i2sd, I2_S, LUT, GEMV, int8, weight-resident, numpy, viable, memory]
---

# Weight-resident I2_S LUT GEMV keeps 4x weight memory vs int8 but loses on speed at B=1/8

## Question

Prior note [[2026-11-07_1055_winsor-clip-absmean-act-vs-int8-bitnetd-boundary]] ruled Winsor-before-absmean on activations **not-viable** and pointed at the weight-side path 05:55 preferred over decode-into-float32. Quant Chief `last_tips.json` `last_run` 2026-11-07T21:09:00Z (~14:09 MT) still wrote pair_A/pair_B **no gap** (roadmap blobs 4efb666/bac6923); no pair item to prefer. Does a **weight-resident** I2_S LUT GEMV (score float32 activations against packed trit weights without expanding weights to a full float32 matrix) recover ≥1.5x speed vs int8-expand+GEMV at B=1/8 while keeping the 4x weight-memory win?

## Method

Synthetic weight tile W shape **(Out=2560, In=2560)**, Gaussian×0.02, seed `202611071455`. Ternarize with plain absmean (`gamma=mean(|W|)+1e-7`, trit=`clip(round(W/gamma),-1,1)`); pack row-major I2_S 2B4T (MSB-first codes 0/1/2, matches `junior_bitnet.i2s.pack`). Int8 baseline: absmax scale, int8-resident, **expand-per-call** to float32 then BLAS GEMM. Activations: Student-t(4)×0.05, B∈{1,8,32}.

**Primary I2_S path (blocked LUT GEMV):** weights stay packed `(Out, In/4)` uint8. For each row-block of 64 (sweep 32/64/128/256/512; 32–64 best at B=1): `np.take` of a 256→4 float32 trit LUT into a preallocated `block×In` scratch, then `y[:,s:e] = (X @ scratch.T) * gamma`. Transient scratch only — never `Out×In` float32. **Alt:** LUT→einsum over 4-wide groups (no BLAS on the weight side). Fair competitor also timed: int8 already expanded to resident f32 (context only).

Error vs ternarized float32 reference GEMV (must be ~0) and vs full-precision W. 3 warm + 7 timed repeats; passes 20/10/4 by B. numpy 2.5.4 only, offline. Local tree `/workspace/JuniorLLM`. Command (scratch `/workspace/bench1455_1107/`):

```
PYTHONPATH=/workspace/JuniorLLM python3 bench.py
```

## Results

Packed I2_S weight bytes are exactly **4.00x** smaller than int8 (1 638 400 vs 6 553 600; f32 would be 26 214 400). Blocked LUT output is exact vs the ternarized reference (max abs ≤1.8e-7). Mae vs full-precision W is the absmean quantization error (~0.028), same order as prior I2_S notes; int8 mae ~0.0007.

| B | I2_S B | int8 B | mem | blocked LUT ms med (min–max) | int8-expand ms med (min–max) | speed vs i8exp | LUT mae vs f32 | int8 mae vs f32 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1638400 | 6553600 | **4.00x** | 3.255 (3.084–3.280) | 2.423 (2.360–2.575) | **0.74x** | 0.02770 | 0.00067 |
| 8 | 1638400 | 6553600 | **4.00x** | 5.915 (5.908–5.962) | 2.504 (2.480–2.714) | **0.42x** | 0.02873 | 0.00070 |
| 32 | 1638400 | 6553600 | **4.00x** | 7.664 (7.612–7.750) | 2.830 (2.758–3.629) | **0.37x** | 0.02919 | 0.00071 |

Context: einsum-fused LUT is slower still (B=1 **15.39 ms**, 0.16x of int8-expand). Int8 already-expanded resident f32 GEMM is 0.10 / 0.36 / 0.62 ms at B=1/8/32 — expand dominates the int8-expand path at small B, yet still beats blocked LUT. Block sweep at B=1: 32→3.19 ms, 64→3.19, 128→5.08, 256→5.78, 512→5.54. Pack matched `junior_bitnet.i2s.pack` on row 0. Exact samples in scratch `results.json`.

## Verdict

**Viable on memory.** Weight-resident I2_S is exactly **4.00x** smaller than int8-resident weights (above the 1.5x bar), with LUT GEMV bit-exact to the ternarized reference and mae vs f32 equal to the absmean quant itself (~0.028 — acceptable, same order as [[2026-11-07_0555_bitnetd-handler-i2s-block-decode-vs-int8-token-batch]]). **Not viable on speed:** blocked byte-LUT + BLAS is only 0.37–0.74x of int8-expand+GEMV at B=1/8/32 (einsum path 0.15–0.16x). Weight-resident does **not** recover the ≥1.5x speed the question asked for; the efficiency win remains storage/resident bytes, not matvec wall time on this numpy box. Keep I2_S weights packed for the 4x mem win; do not expect the LUT GEMV itself to beat int8-expand. Bounded to Out=In=2560 Student-t activations, block=64, shared box.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-07T21:09:00Z (~14:09 MT) wrote pair_A/pair_B **no gap**. Roadmap blobs still JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`. JuniorLLM tip before this push: [120d831](https://github.com/cloudcover95/JuniorLLM/commit/120d831) (prior vault note).

## Next question

Does a **bitplane** (+1/−1 mask select) weight-resident GEMV close the speed gap to int8-expand+GEMV at B=1/8 while matching I2_S’s 4x weight-memory win — i.e. the non-LUT ternary alternative after blocked byte-LUT lost on speed?

close: 7e3c43b49d7443c4 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-07_1055)
