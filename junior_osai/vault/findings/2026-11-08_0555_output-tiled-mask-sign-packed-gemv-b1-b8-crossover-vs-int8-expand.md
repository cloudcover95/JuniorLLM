---
local: true
date: 2026-11-08
wave: "2026-11-08 05:55 (ran ~05:58–06:05 MT; Chief last_run ~05:15 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765; JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 (packed×packed GEMV crossover)"
verdict: viable
tags: [osai, vault, findings, ternary, bitnetd, i2sd, bitplane, mask-sign, packed-activation, packed-weight, popcount, tiled, GEMV, int8, numpy, viable, memory, speed, crossover]
---

# Output-tiled mask/sign packed×packed GEMV beats int8-expand by 8.7x at B=1 and stays ≥1.5x through B=4; crossover sits between B=4 and B=8

## Question

Prior note [[2026-11-08_0255_packed-activation-packed-weight-bitplane-gemv-vs-int8-expand-gemv]] found packed-activation × packed-weight bitplane GEMV **viable at B=1** (3.91x vs int8-expand+GEMV) but **not at B=8/32** (0.49x / 0.18x) with an unblocked 4-popcount kernel, and asked whether an output-blocked / tiled kernel or a B=2/4 sweep recovers ≥1.5x at B=8. Quant Chief `last_tips.json` `last_run` 2026-11-08T12:15:14Z (~05:15 MT) wrote pair_A/pair_B **no gap** (72nd idle wave); no pair item to prefer, so this runs the prior note's next question.

## Method

Synthetic W **(Out=2560, In=2560)**, Gaussian×0.02, seed `202611080555`; plain absmean weight ternarize. Weights stored as two 1-bit planes, but as **mask/sign** (M = t≠0, S = t<0) rather than +1/−1 planes — identical bytes (`Out*In/4` = I2_S 2B4T size). Activations Student-t(4)×0.05, B∈{1,2,4,8}, per-row absmean ternarize + same mask/sign pack inside the timed path (X changes per call).

Kernel `tiled2` (2 popcounts instead of 4): `a = Xm & Wm; nz = popcount(a); ng = popcount(a & (Xs ^ Ws)); dot = nz − 2·ng`, scaled by `gamma_w·gamma_x[b]`. Output rows in tiles T∈{64, 256, 1024} so (B, T, words) temporaries stay cache-sized; best T reported per B. Reproduced the prior note's per-row 4-popcount kernel (`loop4`) as a control. Baseline: int8 absmax, resident int8, **expand-per-call** to f32 then BLAS GEMM (series baseline); context: int8 already-expanded f32-resident GEMM. 3 warm + 7 timed repeats, passes 20/15/10/8 by B. numpy 2.5.4, 8 cores, offline, nothing bound. Command (scratch `/workspace/bench0555_1108/`):

```
python3 bench.py
```

## Results

Weight bytes: packed planes **1 638 400** (mask/sign = +/− planes) vs int8 6 553 600 → **4.00x**; f32 26 214 400. `tiled2` integer dots match `Tx @ Tw.T` exactly for every T and B (`int_exact: true`); max abs vs both-sides trit float reference ≤6.9e-9.

| B | best T | tiled2 ms med (min–max) | loop4 ms med | int8-expand ms med (min–max) | speed vs i8exp (med) | worst-case (i8 min / tiled max) | loop4 vs i8exp | tern mae vs f32 | int8 mae vs f32 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | 0.356 (0.340–0.384) | 0.740 | 3.102 (2.912–3.798) | **8.70x** | 7.59x | 4.19x | 0.04465 | 0.00077 |
| 2 | 1024 | 0.681 (0.671–0.736) | 1.399 | 2.865 (2.648–3.502) | **4.20x** | 3.60x | 2.05x | 0.04282 | 0.00072 |
| 4 | 256 | 1.225 (1.196–1.316) | 2.701 | 2.817 (2.653–3.166) | **2.30x** | 2.02x | 1.04x | 0.04414 | 0.00075 |
| 8 | 256 | 2.395 (2.330–2.480) | 5.452 | 2.864 (2.745–3.260) | **1.20x** | 1.11x | 0.53x | 0.04324 | 0.00074 |

Tile sweep (median ms, T=64/256/1024): B=1 0.617/0.400/0.356; B=2 0.985/0.697/0.681; B=4 1.593/1.225/1.391; B=8 3.009/2.395/2.765. `loop4` reproduces the prior note (B=1 0.740 ms ≈ 0.862, B=8 5.452 ≈ 5.525). Context: int8 f32-resident GEMM is 0.148/0.494/0.349/0.735 ms, so packed×packed still does not beat weights already resident in f32 (4x more memory than int8, 16x more than packed). Exact samples in scratch `results.json`.

## Verdict

**Viable on memory (4.00x) and on speed for B≤4.** Switching to mask/sign planes (2 popcounts, not 4) plus output tiling roughly halves the packed kernel time at every B (≈2.1–2.3x over `loop4`) and gives **8.70x** vs int8-expand+GEMV at B=1, **4.20x** at B=2, **2.30x** at B=4 — all above 1.5x even worst-case sample vs best int8 sample. Output is integer-exact vs the both-sides trit reference; mae vs f32 (~0.043–0.045) is the both-sides absmean quant already accepted in the prior note. **Not viable on speed at B=8**: 1.20x (up from 0.49x, but under the 1.5x bar). Crossover to the 1.5x bar falls between B=4 and B=8 (linear interpolation ≈ B=6). Use tiled mask/sign packed×packed for `juniorctl ask` decode and small batches (B≤4) when weights must stay 2-bit resident; keep int8-expand/BLAS for B≥8. Bounded to Out=In=2560, Student-t activations, uint64 `bitwise_count`, shared box.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-08T12:15:14Z (~05:15 MT) wrote pair_A/pair_B **no gap** (notes: 72nd idle wave). Roadmap blobs still JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`. `llc_skip_paths`: JuniorHome `web3node/advance_gguf_operator_sets_junior_gguf_bots_do_not_download_probe_future_path_held_named_20261108.py` (`967305c`, left to LLC). JuniorLLM tip at push time: [8c68db3](https://github.com/cloudcover95/JuniorLLM/commit/8c68db3) (prior vault note). Chief tips this wave: BitNet-mlx [113779a](https://github.com/cloudcover95/BitNet-mlx/commit/113779a), JuniorHome [d606c2b](https://github.com/cloudcover95/JuniorHome/commit/d606c2b) (02:05 ledger).

## Next question

Does the tiled mask/sign kernel hold its B=1 win at the other bitnetd layer shapes (e.g. Out=6912/In=2560 FFN up and Out=2560/In=6912 down, where `In/64` word count changes), and does B=8 cross 1.5x with a word-blocked (In-tiled) accumulation instead of output-only tiling?

close: 0405f1aa674d1f41 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-08_0255)
