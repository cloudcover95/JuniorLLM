---
local: true
date: 2026-11-09
wave: "2026-11-09 05:55 (ran ~06:00–06:10 MT; Chief last_run ~05:08 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, layer, end-to-end, amdahl, mask-sign, packed-activation, error, int8, f32, numpy, not-viable]
---

# A full bitnetd layer step keeps the packed×packed speed (not Amdahl-limited) but ternary activations wreck the layer output, and f32-resident is still 2x faster

## Question

Prior note [[2026-11-09_0255_tiled-mask-sign-packed-gemv-bitnetd-ffn-shapes-b1-b8-vs-int8-expand]] found output-tiled mask/sign packed×packed GEMV 12–14x faster than int8-expand+GEMV per FFN projection at B=1, and asked whether one whole bitnetd decode-layer step (q/k/v/o 2560×2560, ffn gate/up 6912×2560, down 2560×6912) keeps a ≥1.5x end-to-end win once activation re-ternarize/pack between projections and the f32 RMSNorm / relu² / residual glue are included. Quant Chief `last_tips.json` `last_run` 2026-11-09T12:08:00Z (~05:08 MT) wrote pair_A/pair_B **no gap** (75th idle wave), so this runs the prior note's next question. Two extra checks were added because they decide the answer: the layer output error, and an f32-resident baseline.

## Method

Synthetic layer, seed `202611090555`, each projection Gaussian×0.02 f32. Ternary path: absmean weight trits stored as mask/sign 1-bit planes; activations per-row absmean ternarized and packed once per distinct input (shared for q/k/v and for gate/up, so 4 packs per layer), then the prior note's output-tiled 2-popcount kernel, T∈{256,1024}, best T kept. Glue identical in every path: RMSNorm, o consumes v (softmax/KV left out on purpose), residual, RMSNorm, relu²(gate)·up, sub-RMSNorm, down, residual. Baselines: int8 absmax weights expanded to f32 per call then BLAS (the series baseline), and f32 weights resident (BLAS, 16x the packed bytes). Error is measured on the layer's residual update (out − x) against an f64 layer with the original f32 weights, and also against an f64 layer with the *dequantized ternary* weights (so only the activation-quant error remains). Inputs Student-t(4)×0.5, B∈{1,8}. 3 warm + 7 timed repeats (6 passes at B=1, 3 at B=8); per-phase split from 10 extra instrumented passes; the whole script was run **twice**. numpy 2.5.4, 8 cores, offline, nothing bound. Command (scratch `/workspace/bench0555_1109/`, inline script `bench.py`):

```
python3 bench.py results_run1.json && python3 bench.py results.json
```

## Results

Weight bytes per layer: packed **19 824 640**, int8 79 298 560 (4.00x), f32 317 194 240 (16.00x).

| B | packed tern ms med (min–max), run1 / run2 | int8-expand ms med, run1 / run2 | f32-resident ms med, run1 / run2 | speed vs int8-expand run1 / run2 (worst) | speed vs f32-resident run1 / run2 (worst) |
|---|---|---|---|---|---|
| 1 | 4.278 (4.218–4.382) / 4.308 (4.126–4.646), T=1024 | 52.79 / 65.38 | 2.219 / 2.282 | **12.34x / 15.18x** (11.60x / 11.81x) | **0.52x / 0.53x** (0.42x / 0.44x) |
| 8 | 29.14 (28.62–29.41) / 31.87 (31.23–32.77), T=256 | 55.62 / 52.14 | 6.480 / 5.356 | 1.91x / 1.64x (1.75x / 1.43x) | 0.22x / 0.17x (0.20x / 0.15x) |

Where the ternary layer time goes (share of instrumented time, run1 / run2): B=1 GEMV **95.5% / 95.6%**, activation ternarize+pack 2.9% / 2.8% (0.13–0.15 ms), f32 glue 1.6% (0.08 ms). B=8 GEMV 98.5%, pack 0.8–0.9%, glue 0.6–0.7%.

Layer residual-update error (identical in both runs, deterministic):

| path | reference | mae | ref rms | cosine |
|---|---|---|---|---|
| int8 W, f32 act | f64, f32 weights | 0.0437 | 1.914 | **0.9996** |
| ternary W, ternary act | f64, f32 weights | 1.4815 | 1.914 | **0.307** |
| ternary W, ternary act | f64, native ternary weights | 0.8926 | 1.203 | **0.437** |

(B=8 matches: cosine 0.297 / 0.426, int8 0.9996.) Full samples in scratch `results_run1.json` / `results.json`.

## Verdict

**Not viable end-to-end.** The speed side is fine and the answer to the Amdahl question is no: GEMV is ~95% of the ternary layer, re-pack plus glue only ~4.5%, and the whole layer stays 12–15x faster than int8-expand at B=1, so the per-projection win carries through. But two things sink it. (1) **Error:** with activations also ternarized, the layer's residual update keeps cosine only **0.44** even against native-ternary weights (0.31 against the f32 weights), while int8 keeps 0.9996; a single projection's ~0.6-of-rms mae (accepted in the prior notes) compounds across four projections in series and the relu² gate. Ternary activations are not an acceptable decode path; bitnetd's BitNet b1.58 contract is ternary weights with **8-bit** activations, so the prior notes' packed×packed (ternary-activation) "viable" verdicts hold for kernel speed only, not for model output. (2) **Speed vs the real resident baseline:** f32-resident BLAS is **~2x faster** at B=1 (0.52–0.53x) and 4.5–6x faster at B=8; the 12–15x number is only against a baseline that re-expands int8 every call. Ternary's real win here stays **memory (4.00x vs int8, 16.00x vs f32)**. Bounded to synthetic Gaussian weights (not a trained ternary model, which would lower the weight-side part of the error), softmax/KV omitted, numpy `bitwise_count`, shared 8-core box.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-09T12:08:00Z (~05:08 MT) wrote pair_A/pair_B **no gap** (roadmap blobs unchanged, JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). `llc_skip_paths`: JuniorHome docs ledger (tip `0cab910`, left to LLC). JuniorLLM tip at push time: [e26d397](https://github.com/cloudcover95/JuniorLLM/commit/e26d397) (prior vault note).

## Next question

At B=1 decode, does ternary-weight × **int8-activation** (W1.58A8, the real bitnetd contract) via weight mask/sign planes and int8 activation bit-slices (8 popcount planes, or an int8 LUT over packed weights) get within 1.5x of f32-resident BLAS speed while keeping the 4x / 16x memory win and cosine ≥ 0.999 on the layer update?

close: e4392abde78ca90c (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-09_0255)
