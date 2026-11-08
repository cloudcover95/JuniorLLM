---
local: true
date: 2026-11-07
wave: "2026-11-07 18:55 (ran ~19:07–19:08 MT; Chief last_tips ~18:08 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 6 / JuniorOS overlay: bitnetd on 127.0.0.1:8765; JuniorHome docs/ROADMAP.md #3 i2sd userspace 127.0.0.1:8767 (weight-resident bitplane mask-select GEMV)"
verdict: viable
tags: [osai, vault, findings, ternary, bitnetd, i2sd, bitplane, mask-select, GEMV, int8, weight-resident, numpy, viable, memory]
---

# Weight-resident bitplane mask-select GEMV keeps 4x weight memory vs int8 but does not close the ≥1.5x speed gap at B=1/8

## Question

Prior note [[2026-11-07_1455_weight-resident-i2s-lut-gemv-vs-int8-expand-gemv]] found weight-resident I2_S LUT GEMV **viable on memory** (exactly 4.00x vs int8) but **not on speed** (blocked byte-LUT 0.37–0.74x of int8-expand+GEMV) and pointed at the non-LUT ternary alternative: bitplane (+1/−1 mask select). Quant Chief `last_tips.json` `last_run` 2026-11-08T01:08:09Z (~18:08 MT) still wrote pair_A/pair_B **no gap** (roadmap blobs 4efb666/bac6923; notes call the 18:05 wave the 70th idle since 2026-10-24); no pair item to prefer. Earlier bitplane notes ([[2026-10-15_2025_bitplane-bench]], [[2026-10-15_2030_native-bitplane]], [[2026-10-16_1455_popcount-bitplane-matvec-vs-fp32]]) compared other setups (mostly vs fp32 / popcount). Does a **weight-resident** bitplane (+1/−1 mask select) GEMV close the speed gap to int8-expand+GEMV at B=1/8 while matching I2_S’s 4x weight-memory win?

## Method

Synthetic weight tile W shape **(Out=2560, In=2560)**, Gaussian×0.02, seed `202611071855`. Ternarize with plain absmean (`gamma=mean(|W|)+1e-7`, trit=`clip(round(W/gamma),-1,1)`). Pack two 1-bit planes for +1 and −1 (zeros are neither) as uint8 `(Out, In/8)` each via `np.packbits`; total packed bytes = `2*Out*In/8` = `Out*In/4` = I2_S 2B4T size. Int8 baseline: absmax scale, int8-resident, **expand-per-call** to float32 then BLAS GEMM. Activations: Student-t(4)×0.05, B∈{1,8,32}.

**Primary bitplane path (blocked unpack + ±1 scratch GEMV):** weights stay two packed planes. For each row-block of 64 (sweep 32/64/128/256/512; 64 best at B=1): `np.unpackbits` pos/neg into a preallocated `block×In` float32 scratch of +1/−1/0, then `y[:,s:e] = (X @ scratch.T) * gamma`. Transient scratch only — never `Out×In` float32. **Alt (context):** row-wise boolean mask-select `y = gamma * (sum(X[:,pos]) - sum(X[:,neg]))` per output row; also blocked two-GEMM `(X @ pos.T) - (X @ neg.T)`. Fair competitor also timed: int8 already expanded to resident f32 (context only). Int8-expand timed before the slow row-wise path so the 75–230 ms Python loop does not pollute the headline medians.

Error vs ternarized float32 reference GEMV (must be ~0) and vs full-precision W. 3 warm + 7 timed repeats; passes 20/10/4 by B. numpy 2.5.4 only, offline. Local tree `/workspace/JuniorLLM`. Command (scratch `/workspace/bench1855_1107/`):

```
PYTHONPATH=/workspace/JuniorLLM python3 bench.py
```

## Results

Packed bitplane weight bytes are exactly **4.00x** smaller than int8 (1 638 400 vs 6 553 600; equals I2_S 2B4T; f32 would be 26 214 400). Blocked ±1 output is exact vs the ternarized reference (max abs ≤1.3e-7). Mae vs full-precision W is the absmean quantization error (~0.028), same order as [[2026-11-07_1455_weight-resident-i2s-lut-gemv-vs-int8-expand-gemv]]; int8 mae ~0.0007.

| B | bitplane B | int8 B | mem | blocked ±1 ms med (min–max) | int8-expand ms med (min–max) | speed vs i8exp | bp mae vs f32 | int8 mae vs f32 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1638400 | 6553600 | **4.00x** | 2.518 (2.505–2.896) | 2.970 (2.871–3.175) | **1.18x** | 0.02804 | 0.00066 |
| 8 | 1638400 | 6553600 | **4.00x** | 7.229 (7.179–7.330) | 4.547 (4.271–4.807) | **0.63x** | 0.02900 | 0.00069 |
| 32 | 1638400 | 6553600 | **4.00x** | 11.447 (11.408–11.517) | 5.699 (5.156–6.345) | **0.50x** | 0.02913 | 0.00068 |

Context: row-wise mask-select is far slower (B=1 **76.93 ms**, 0.039x of int8-expand; B=8/32 163.2 / 227.7 ms). Two-GEMM blocked unpack is 2.471 / 11.636 / 20.149 ms at B=1/8/32 (1.20x / 0.39x / 0.28x) — a hair faster than ±1 scratch at B=1 only, then loses as B grows. Int8 already-expanded resident f32 GEMM is 0.104 / 0.424 / 0.750 ms at B=1/8/32 — expand still dominates the int8-expand path at small B. Block sweep at B=1 (signed path): 32→2.749 ms, 64→2.548, 128→2.806, 256→4.475, 512→4.193. Packed-byte count matched I2_S. Exact samples in scratch `results.json`.

## Verdict

**Viable on memory.** Weight-resident bitplane (+1/−1 planes) is exactly **4.00x** smaller than int8-resident weights (above the 1.5x bar; same packed size as I2_S 2B4T), with blocked unpack GEMV bit-exact to the ternarized reference and mae vs f32 equal to the absmean quant itself (~0.028 — acceptable, same order as [[2026-11-07_1455_weight-resident-i2s-lut-gemv-vs-int8-expand-gemv]]). **Not viable on speed:** blocked ±1 scratch + BLAS is **1.18x** of int8-expand+GEMV at B=1 (below the 1.5x bar) and only **0.63x / 0.50x** at B=8/32. Bitplane did improve on the 14:55 blocked LUT at B=1 (that note’s 0.74x → 1.18x here) and slightly beat int8-expand at B=1, but it does **not** close the ≥1.5x speed gap the question asked for at B=1/8. The efficiency win remains storage/resident bytes, not a new matvec wall-time win on this numpy box. Keep 2-bit packed weights for the 4x mem win; do not expect weight-resident bitplane mask-select itself to beat int8-expand by ≥1.5x. Bounded to Out=In=2560 Student-t activations, block=64, shared box.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-08T01:08:09Z (~18:08 MT) wrote pair_A/pair_B **no gap** (notes: 70th idle wave at 18:05). Roadmap blobs still JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`. No local 16:05 or 18:05 ledger clone; searched `/tmp` and `/workspace` (`last_tips.json` at `/workspace/quant-chief/last_tips.json` only). JuniorLLM tip before this push: [fbbea56](https://github.com/cloudcover95/JuniorLLM/commit/fbbea56) (prior vault note).

## Next question

Does a **packed-activation × packed-weight** bitplane GEMV (both sides stay bitpacked; no float32 expand of W or X) beat int8-expand+GEMV at B=1/8 while keeping the 4x weight-memory win — i.e. after weight-only bitplane mask-select won memory and only a 1.18x (not ≥1.5x) speed edge at B=1? Distinct from [[2026-10-16_1455_popcount-bitplane-matvec-vs-fp32]] (activation-quant popcount vs fp32, not vs this series’ int8-expand baseline).

close: 3f7d84bab220577f (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-07_1455)
