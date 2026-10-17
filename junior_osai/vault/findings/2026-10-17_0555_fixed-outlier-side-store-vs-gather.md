---
local: true
date: 2026-10-17T05:55:00-06:00
wave: "2026-10-17 05:05"
roadmap_item: "i2sd userspace 127.0.0.1:8767 no docker.sock"
roadmap_ref: "JuniorHome docs/ROADMAP.md #3 (Trit Pack's pair_A item this wave, per its commit d5f5985)"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnet, i2sd, activation-quant, outlier-split, side-store, calibration, popcount, bitplane, benchmark]
---

# Fixed calibrated outlier-column side store vs per-token gather for B=1 ternary popcount matvec

- Prev: [[2026-10-17_0255_outlier-split-act-quant-vs-absmax]]
- Answers: that note's next question (a small column-major copy of the ternary weights for a fixed, calibrated set of outlier channels, instead of gathering columns from the packed masks per token)
- Roadmap: i2sd userspace 127.0.0.1:8767 no docker.sock (JuniorHome docs/ROADMAP.md #3)

## Question
The 02:55 note found that splitting the top 0.5% |x| channels into an fp32 side matvec gets int8 under 1% error, but gathering those columns out of the row-packed +1/-1 masks costs 0.5 to 0.95 ms per token. If i2sd instead calibrates a fixed channel set S once (top p of mean |x| over 64 tokens) and stores T[:, S] as its own column-major int8 or fp32 array, how much does the side cost drop, how many extra bytes does it add per layer, and how much error comes back when a share of each token's outlier channels drifts outside S?

## Method
- Shapes (out x in): 4096x4096 and 2560x6912. T drawn from {-1, 0, +1} with p = 0.3/0.4/0.3, packed into two uint64 bitmasks (4 bits/weight). Seed 20261017_0555 mod 2^32.
- Activations: standard normal plus 0.5% outlier channels scaled by 20 x lognormal(0, 0.25). Per token, a fraction `drift` of the outlier channels is drawn fresh at random and the rest come from one fixed set; drift = 0, 0.1, 0.25, 0.5, 1.0.
- Calibration: 64 tokens at the same drift, S = top p mean |x| channels, p = 0.5% and 1%. Test: 16 fresh tokens per drift.
- Paths (all k-bit absmax activations through the same popcount bitplane matvec, k = 6 and 8): plain absmax; dynamic split (per-token top-p, columns gathered from the masks, as at 02:55); fixed split (zero x[S], side matvec over the stored T[:, S] in fp32 or int8 cast per token; the two agree to rtol 1e-5).
- Error: relative L2 against float64 T @ x, median and max over the 16 test tokens. Timing: 31 repeats after a warm-up, median and IQR, one drift-0.25 token, BLAS at 1 and 8 threads. Raw JSON in `/workspace/tbench/r0555/threads1.json` and `threads8.json` (box-local, not pushed). Peak RSS 341 MB.

```
cd /workspace/tbench
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 fixed_side_store_bench.py 31 > r0555/threads1.json
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 fixed_side_store_bench.py 31 > r0555/threads8.json
```

Core of `fixed_side_store_bench.py` (numpy 2.5.4, `np.bitwise_count`; quant, popcount_matvec and gather_cols are unchanged from the 02:55 script):

```python
def dynamic(P, N, rs, x, k, m, w8):          # 02:55 path: per-token top-m, gather from packed masks
    idx = np.argpartition(np.abs(x), -m)[-m:]; xi = x.copy(); xs = x[idx]; xi[idx] = 0.0
    q, s = quant(xi, k); return popcount_matvec(P, N, rs, q, k, w8) * s + gather_cols(P, N, idx) @ xs
def fixed_f32(P, N, rs, x, k, S, WsT, w8):   # WsT = ascontiguousarray(Wf[:, S].T), built once at calibration
    xi = x.copy(); xs = x[S]; xi[S] = 0.0
    q, s = quant(xi, k); return popcount_matvec(P, N, rs, q, k, w8) * s + xs @ WsT
def fixed_i8(P, N, rs, x, k, S, Ws8T, w8):   # Ws8T = ascontiguousarray(T[:, S].T) int8, cast per token
    xi = x.copy(); xs = x[S]; xi[S] = 0.0
    q, s = quant(xi, k); return popcount_matvec(P, N, rs, q, k, w8) * s + xs @ Ws8T.astype(np.float32)
# calibration: score = abs(stack(64 tokens)).mean(0); S = sort(argpartition(score, -m)[-m:]), m = ceil(p * C)
```

## Results

Side cost only (ms, median, 1 thread; 8-thread numbers within 0.2 ms of these):

| shape | p | m cols | dynamic gather + side | fixed fp32 side | fixed int8 side | gather / fixed fp32 | gather / fixed int8 |
|---|---|---|---|---|---|---|---|
| 4096x4096 | 0.5% | 21 | 0.851 | 0.0064 | 0.0135 | 133x | 63x |
| 4096x4096 | 1% | 41 | 1.647 | 0.0059 | 0.0188 | 279x | 88x |
| 2560x6912 | 0.5% | 35 | 0.511 | 0.0069 | 0.0180 | 74x | 28x |
| 2560x6912 | 1% | 70 | 1.199 | 0.0093 | 0.0311 | 129x | 39x |

Extra bytes per layer for the side store (packed masks are 4.19 MB and 4.42 MB, fp16 weights 33.6 MB and 35.4 MB):

| shape | p | int8 store | fp32 store | int8 / masks |
|---|---|---|---|---|
| 4096x4096 | 0.5% | 86,016 B | 344,064 B | 2.0% |
| 4096x4096 | 1% | 167,936 B | 671,744 B | 4.0% |
| 2560x6912 | 0.5% | 89,600 B | 358,400 B | 2.0% |
| 2560x6912 | 1% | 179,200 B | 716,800 B | 4.1% |

Whole B=1 matvec, k = 8 (ms, median [IQR]):

| shape | fp32 BLAS 1 thr | fp32 BLAS 8 thr | plain absmax | dynamic p0.5% | fixed fp32 p0.5% | fixed int8 p0.5% | dynamic p1% | fixed fp32 p1% |
|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 2.150 [2.09-2.23] | 0.291 [0.29-0.30] | 6.727 | 7.758 [7.52-8.12] | 6.695 [6.64-6.96] | 6.712 | 8.394 | 6.735 |
| 2560x6912 | 2.407 [2.32-2.52] | 0.351 [0.34-0.36] | 6.886 | 7.362 [7.24-7.80] | 6.927 [6.71-7.41] | 7.052 | 8.017 | 6.940 |

Error, relative L2 (median / max over 16 tokens), p = 0.5%:

| shape | drift | k | plain | dynamic | fixed |
|---|---|---|---|---|---|
| 4096x4096 | 0 | 8 | 6.16% / 8.71% | 0.42% / 0.58% | 0.45% / 0.70% |
| 4096x4096 | 0.1 | 8 | 5.76% / 10.6% | 0.42% / 0.56% | 2.40% / 10.6% |
| 4096x4096 | 0.25 | 8 | 5.78% / 7.40% | 0.43% / 0.51% | 4.17% / 7.09% |
| 4096x4096 | 0.5 | 8 | 6.30% / 11.6% | 0.44% / 0.49% | 5.49% / 8.08% |
| 4096x4096 | 1.0 | 8 | 6.17% / 9.94% | 0.41% / 0.52% | 6.14% / 9.91% |
| 4096x4096 | 0 | 6 | 25.0% / 34.5% | 1.72% / 2.28% | 1.83% / 2.85% |
| 2560x6912 | 0 | 8 | 7.44% / 9.97% | 0.41% / 0.50% | 0.46% / 0.58% |
| 2560x6912 | 0.1 | 8 | 6.80% / 8.92% | 0.42% / 0.50% | 2.89% / 5.97% |
| 2560x6912 | 0.25 | 8 | 6.74% / 8.21% | 0.43% / 0.48% | 4.84% / 7.69% |
| 2560x6912 | 0.5 | 8 | 5.96% / 9.43% | 0.45% / 0.51% | 5.41% / 8.94% |
| 2560x6912 | 1.0 | 8 | 7.33% / 9.87% | 0.43% / 0.49% | 7.31% / 9.84% |
| 2560x6912 | 0 | 6 | 30.4% / 38.4% | 1.66% / 2.06% | 1.90% / 2.34% |

Calibration recall of the fixed outlier set was 100% for drift 0 to 0.5 (the fixed channels still have the highest mean |x|) and 0% at drift 1.0, as expected. Going from p = 0.5% to 1% changes the fixed-path error by under 0.01 percentage points at every drift, so a bigger S does not catch drifted channels.

## Verdict
**Not viable on speed; the side store is a real fix for the gather cost, but only when outlier channels do not move.** Storing T[:, S] once makes the side matvec 74x to 279x cheaper than gathering from the packed masks (fp32 store) or 28x to 88x cheaper (int8 store), for 2% to 4% extra bytes over the masks with int8. With no drift the fixed path matches the dynamic split on error (int8 activations 0.45% vs 0.42%), and it runs at the same speed as plain absmax (6.70 vs 6.73 ms), so the 02:55 split's 0.5 to 1.6 ms penalty is gone.

Two things keep it from being a win. First, the whole ternary path is still 2.9x to 3.1x slower than 1-thread fp32 BLAS and 19x to 23x slower than 8-thread BLAS, because the popcount bitplane matvec itself (about 6.7 ms) dominates; removing the side cost only improves the split path by 1.06x to 1.25x. Second, the error is fragile: with just 10% of a token's outlier channels outside S, median int8 error jumps from 0.45% to 2.4% to 2.9% (max 10.6%), and at 25% drift it is 4.2% to 4.8%, close to plain absmax. One uncaptured outlier sets the absmax scale for the whole vector. Int6 still never reaches 1% (1.83% to 1.90% at zero drift).

For i2sd: keep a fixed int8 side store for calibrated outlier channels (it is nearly free in bytes and time), but pair it with a per-token check for uncaptured outliers rather than trusting the calibration alone. The weight-memory win (16x vs fp32, 8x vs fp16 for these masks) is unchanged and already recorded; latency remains the blocker.

## Ops check
`python3 scripts/terraform_lean_prod.py code` and `... llm` in local `/workspace/JuniorLLM` (offline, blender off): both exited 0 with `"ok": true`, `"bind": "127.0.0.1"`, `"bpy": false`, `"model_pull": false`, `"launch": false`.

## This wave's pair files (2026-10-17 05:05)
- Trit Pack: [JuniorLLM scripts/i2sd_wire_jtr1_key.py @ d5f5985](https://github.com/cloudcover95/JuniorLLM/blob/d5f5985/scripts/i2sd_wire_jtr1_key.py) (#3 i2sd, Chief pair A per its commit)
- Winsor: no new commit on JuniorQuant or JuniorStock on GitHub as of 06:05 MT
- Side-Commit: [web3node gguf_header_receipt.py @ 558f474](https://github.com/cloudcover95/web3node/blob/558f474/gguf_header_receipt.py) (#2 GGUF)
- Handshake & Ports: [JuniorLLM ports/gguf_disk_gate_port_map.py @ d06ee93](https://github.com/cloudcover95/JuniorLLM/blob/d06ee93/ports/gguf_disk_gate_port_map.py) (BETA_TO_OS JuniorOS overlay, GGUF-on-disk gate)

## Next question
Fixed side store plus a residual guard: after zeroing x[S], flag any remaining channel with |x| above a threshold (for example 8x the token's RMS) and either gather just those few columns from the masks or give them their own scale. How often does the guard fire at drift 0.1 to 0.5, what does it cost per token compared with the 0.5 to 1.6 ms full gather, and does it bring int8 back under 1% at every drift?

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-17_0255_outlier-split-act-quant-vs-absmax.md.

close: 209c1daecce4c9e4
