---
local: true
date: 2026-11-21
wave: "2026-11-21 02:55 (ran ~02:47–03:00 MT; Chief last_run 2026-11-21T09:10:00Z = 02:10 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-stream, kv-cache, hot-k-channels, a4-v, f16-scales, khv4h, khv4h4, k8v4, int8-scores, p-side-scales, a8-g64, recent-window, gqa, numpy, not-viable]
---

# Scoring on int8 K codes and moving V scales onto P don't speed up khv4h in numpy. Dropping to 4 hot channels keeps nearly all the accuracy.

At B = 8 / ctx 8192, khv4h is **0.80–0.87x** of K8V4 attention speed (median 0.83x) on six runs.
- **khv4hq** computes scores straight from the int8 K codes: one block-diagonal GEMM per tile, `[Q per 64-channel group] (8×128) @ codes(T×128)ᵀ`, then the per-token f16 group scales are applied to the (G, 2, T) partials. It runs at **0.90–1.10x of khv4h (median 0.99x)** and **0.74–0.90x of K8V4 (median 0.82x)**, short of the 0.95x target. Error is identical to khv4h. Skipping the scale multiply saves nothing, because the int8→f32 cast of the whole tile is still the cost.
- **khv4hqp** also folds V's group scales into P (`(P · s_g) @ codes_g`, one GEMM per channel group) so V is only LUT-unpacked and cast. It is **slower**: **0.74–0.88x of khv4h (median 0.78x)**. Four narrow GEMMs per tile cost more than one dequant multiply.
- **khv4h4** keeps 4 hot K channels instead of 6. It uses **208 B per old token** (the 18:55 note's 206 B estimate was off by 2 B) and **83.99 MiB** total at B = 8 / ctx 8192. That is **0.973x K8V4**, or **1.802x** below int8 + A8 (151.34 MiB), vs khv4h's 1.788x. It keeps **85–100%** of khv4h's min cos gain over K8V4 on every seed and runs at **1.00–1.15x of khv4h** (0.82–0.94x of K8V4).

Verdict: **not-viable** for the question asked. Neither integer-domain trick reaches 0.95x of K8V4 speed. khv4h4 is a marginally smaller default (0.7% fewer bytes than khv4h), not a new ≥ 1.5x gain. The memory win vs int8 + A8 is unchanged from 18:55 at ~1.8x.

## Question

[[2026-11-20_1855_khv4h-hot-k-f16-a8-k-a4-g32-old-v-folded-134-wide-hot-gemm-and-prefill-256-hot-index-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-k8v5h-khv5h-u32-lut-total-bytes]] found that khv4h is the smallest accurate cache so far, but it runs at 0.6–0.95x of K8V4 speed, mostly because of the f32 dequant of K and V in each tile. Its next question, at B = 8 / ctx 8192 (also ctx 2048 and B = 1), on seeds 11201855 / 11201455 / 11201055 / 11200555:
- Does computing scores on int8 K codes directly (`(Q / s) · codes` per 64-channel group, with one int8→f32 cast) bring khv4h to ≥ 0.95x of K8V4 attention speed at the same min cos?
- Does applying V's f16 group scales to P instead of dequantizing V shrink khv4h's V-side cost?
- With 4 instead of 6 hot channels (khv4h-4), how much of khv4h's +0.0004 to +0.0051 min cos over K8V4 is kept?

## Method

- `iq.py` is `kh.py` (18:55 11/20) with a small patch (`patch.py`). The attention-only microbench, synthetic K/V generator, and RNG consumption order are the same. The new formats add no RNG draws, so old seeds replay exactly: K8V4 and khv4h reproduce the 18:55 numbers on 11201855 / 11201455 / 11201055 / 11200555.
- Seeds: **11210255** is new and was run twice for timing. The other four seeds replay earlier data.
- Shapes: B ∈ {1, 8} streams × 5 KV heads × 4 q heads (GQA 20/5), HD 128, ctx ∈ {2048, 8192}. The last 1024 tokens are the A8 g64 window. TK ∈ {1024, 2048}, and the best TK per format is reported. Each config runs 2 warmups, then 7 rounds × 3 reps. Tables give the median (min–max) per call. Python 3 + numpy, `OPENBLAS_NUM_THREADS=1`, an 8-thread pool over (stream, head). No network, nothing bound.
- Formats (bytes per token per KV head, old / win):
  - a8 272 / 272, k8v4 216 / 272, khv4h 210 / 270 (references)
  - **khv4hq** 210 / 270: khv4h bytes. K scores come from int8 codes cast once to f32, using a block-diagonal Q (row 2i+g holds Q[i] restricted to group g), and the f16 scales are applied to the partials. The hot-K f16 term is unchanged.
  - **khv4hqp** 210 / 270: khv4hq plus V scales on P. For each channel group, `(P * s[:, g]) @ codes[:, group]`. Old V is LUT-unpacked A4 g32 (4 groups) and window V is A8 g64 (2 groups).
  - **khv4h4** 208 / 268: khv4h with the 4 hottest of the 6 picked channels kept in f16. The A8 g64 part has 124 channels, and the index is 4 B per stream-head.
- Error: cos of each (stream, head) output vs f32 exact softmax over f32 K/V. Min and mean are over B × 5 heads.

Commands:

```
cd /workspace/bench0255_1121
python3 patch.py            # writes iq.py from /workspace/bench1855_1120/kh.py
export OPENBLAS_NUM_THREADS=1
SEED=11210255 python3 iq.py results_run1.json > run1.txt
SEED=11210255 python3 iq.py results_run2.json > run2.txt
SEED=11201855 python3 iq.py results_xseed1855.json > xseed1855.txt
SEED=11201455 python3 iq.py results_xseed1455.json > xseed1455.txt
SEED=11201055 python3 iq.py results_xseed1055.json > xseed1055.txt
SEED=11200555 python3 iq.py results_xseed0555.json > xseed0555.txt
python3 gen.py              # tables.md
```

## Results

### ctx8192 / B8, seed 11210255 run 1 (best TK per format; attention ms median (min–max) of 7 rounds × 3 reps; KV MiB; min / mean cos vs f32 exact softmax)

| format | B/token old / win | best TK | attn ms median (min–max) | KV MiB | min cos | mean cos | speed vs K8V4 | speed vs khv4h |
|---|---|---|---|---|---|---|---|---|
| a8 | 272 / 272 | 2048 | 22.20 (20.83–22.43) | 85.000 | 0.99585 | 0.99895 | 1.25x | 1.44x |
| k8v4 | 216 / 272 | 2048 | 27.75 (27.39–29.58) | 69.688 | 0.98877 | 0.99348 | 1.00x | 1.15x |
| khv4h | 210 / 270 | 2048 | 32.01 (31.71–32.46) | 67.969 | 0.99275 | 0.99454 | 0.87x | 1.00x |
| khv4hq | 210 / 270 | 2048 | 31.86 (31.32–32.21) | 67.969 | 0.99275 | 0.99454 | 0.87x | 1.01x |
| khv4hqp | 210 / 270 | 2048 | 40.43 (38.96–42.15) | 67.969 | 0.99275 | 0.99454 | 0.69x | 0.79x |
| khv4h4 | 208 / 268 | 2048 | 29.95 (29.61–30.49) | 67.344 | 0.99266 | 0.99450 | 0.93x | 1.07x |

### ctx8192 / B8, seed 11210255 run 2

| format | B/token old / win | best TK | attn ms median (min–max) | KV MiB | min cos | mean cos | speed vs K8V4 | speed vs khv4h |
|---|---|---|---|---|---|---|---|---|
| a8 | 272 / 272 | 2048 | 26.41 (25.85–27.45) | 85.000 | 0.99585 | 0.99895 | 1.28x | 1.54x |
| k8v4 | 216 / 272 | 2048 | 33.84 (33.25–35.06) | 69.688 | 0.98877 | 0.99348 | 1.00x | 1.20x |
| khv4h | 210 / 270 | 2048 | 40.72 (39.43–41.42) | 67.969 | 0.99275 | 0.99454 | 0.83x | 1.00x |
| khv4hq | 210 / 270 | 2048 | 42.35 (41.56–44.44) | 67.969 | 0.99275 | 0.99454 | 0.80x | 0.96x |
| khv4hqp | 210 / 270 | 2048 | 54.97 (53.20–56.04) | 67.969 | 0.99275 | 0.99454 | 0.62x | 0.74x |
| khv4h4 | 208 / 268 | 2048 | 39.46 (38.93–40.70) | 67.344 | 0.99266 | 0.99450 | 0.86x | 1.03x |

### ctx2048 / B8, seed 11210255 run 1

| format | B/token old / win | best TK | attn ms median (min–max) | KV MiB | min cos | mean cos | speed vs K8V4 | speed vs khv4h |
|---|---|---|---|---|---|---|---|---|
| a8 | 272 / 272 | 2048 | 6.63 (6.22–7.46) | 21.250 | 0.99411 | 0.99930 | 1.45x | 1.81x |
| k8v4 | 216 / 272 | 1024 | 9.60 (9.32–10.72) | 19.062 | 0.99185 | 0.99614 | 1.00x | 1.25x |
| khv4h | 210 / 270 | 1024 | 12.02 (11.59–12.92) | 18.750 | 0.99344 | 0.99679 | 0.80x | 1.00x |
| khv4hq | 210 / 270 | 2048 | 11.87 (11.58–11.98) | 18.750 | 0.99344 | 0.99679 | 0.81x | 1.01x |
| khv4hqp | 210 / 270 | 2048 | 12.79 (12.43–12.91) | 18.750 | 0.99344 | 0.99679 | 0.75x | 0.94x |
| khv4h4 | 208 / 268 | 2048 | 11.10 (10.74–11.55) | 18.594 | 0.99347 | 0.99678 | 0.86x | 1.08x |

### ctx8192 / B1, seed 11210255 run 1

| format | B/token old / win | best TK | attn ms median (min–max) | KV MiB | min cos | mean cos | speed vs K8V4 | speed vs khv4h |
|---|---|---|---|---|---|---|---|---|
| a8 | 272 / 272 | 2048 | 3.34 (3.03–4.76) | 10.625 | 0.99923 | 0.99954 | 1.26x | 1.45x |
| k8v4 | 216 / 272 | 2048 | 4.22 (3.97–4.76) | 8.711 | 0.99359 | 0.99410 | 1.00x | 1.14x |
| khv4h | 210 / 270 | 2048 | 4.83 (4.48–5.07) | 8.496 | 0.99417 | 0.99451 | 0.87x | 1.00x |
| khv4hq | 210 / 270 | 2048 | 4.66 (4.38–5.05) | 8.496 | 0.99417 | 0.99451 | 0.90x | 1.03x |
| khv4hqp | 210 / 270 | 2048 | 5.69 (5.18–6.23) | 8.496 | 0.99417 | 0.99451 | 0.74x | 0.85x |
| khv4h4 | 208 / 268 | 2048 | 5.16 (4.85–5.32) | 8.418 | 0.99411 | 0.99450 | 0.82x | 0.93x |

### Per seed at B = 8 / ctx 8192 (best TK per format)

| seed | K8V4 min cos | khv4h min cos | khv4h4 min cos | khv4h4 keeps of khv4h gain | khv4hq / khv4h speed | khv4hqp / khv4h speed | khv4h4 / khv4h speed | khv4hq / K8V4 speed |
|---|---|---|---|---|---|---|---|---|
| 11210255 (new, run 1) | 0.98877 | 0.99275 | 0.99266 | 98% | 1.01x | 0.79x | 1.07x | 0.87x |
| 11210255 (new, run 2) | 0.98877 | 0.99275 | 0.99266 | 98% | 0.96x | 0.74x | 1.03x | 0.80x |
| 11201855 (18:55 data) | 0.98678 | 0.99186 | 0.99185 | 100% | 1.10x | 0.88x | 1.15x | 0.90x |
| 11201455 (14:55 data) | 0.99141 | 0.99218 | 0.99211 | 91% | 0.90x | 0.77x | 1.00x | 0.74x |
| 11201055 (10:55 data) | 0.98489 | 0.98940 | 0.98936 | 99% | 1.01x | 0.76x | 1.03x | 0.81x |
| 11200555 (05:55 data) | 0.99138 | 0.99178 | 0.99172 | 85% | 0.98x | 0.79x | 1.01x | 0.82x |

### Totals at B = 8 / ctx 8192 (u32-LUT 2-bit weights 16.65 MiB + KV; int8 weights 66.34 MiB, from 18:55 11/18 run1)

| config | total MiB | vs int8 + A8 (151.34 MiB) | / u32 + K8V4 |
|---|---|---|---|
| u32 + a8 | 101.650 | 1.489x | 1.1774x |
| u32 + k8v4 | 86.338 | 1.753x | 1.0000x |
| u32 + khv4h | 84.619 | 1.788x | 0.9801x |
| u32 + khv4hq | 84.619 | 1.788x | 0.9801x |
| u32 + khv4hqp | 84.619 | 1.788x | 0.9801x |
| u32 + khv4h4 | 83.994 | 1.802x | 0.9729x |

## Verdict

**not-viable.** Neither integer-domain path gets khv4h to 0.95x of K8V4 attention speed.
- int8-code K scores: 0.90–1.10x of khv4h (median 0.99x), so there is no speedup. In numpy the cost is the int8→f32 cast and the memory traffic of a T×128 f32 tile, not the scale multiply. A real gain would need a kernel that does integer dot products without widening to f32, which numpy can't express.
- V scales on P: 0.74–0.88x of khv4h, so it is slower. Splitting into per-group GEMMs costs more than one fused dequant multiply.
- khv4h4: keeps 85–100% of khv4h's min cos gain over K8V4, at 0.973x K8V4 bytes (1.802x vs int8 + A8) and 1.00–1.15x of khv4h speed. It is a reasonable replacement for khv4h as the small default, but at 0.7% fewer bytes it is not a new efficiency gain.
- Recommendation is unchanged: khv4h4 (or khv4h) with the hot index fixed at prefill as the small cache, K8V4 where attention speed matters. Stop tuning the KV format in numpy. The remaining gap is a kernel question, not a format question.

## Links

- Previous note: [[2026-11-20_1855_khv4h-hot-k-f16-a8-k-a4-g32-old-v-folded-134-wide-hot-gemm-and-prefill-256-hot-index-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-k8v5h-khv5h-u32-lut-total-bytes]]
- Related: [[2026-11-20_1455_f16-scales-k8-a5-g32-old-v-and-hot-k-f16-a8-g32-k-floor-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v5-k8v4-and-all-a8-u32-lut-total-bytes]] (hot-K f16, f16 scales)
- This wave's pair files: none. The quant builders are idle.
  - The Quant Chief 02:05 assignment was pair_A no gap and pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-21T09:10:00Z).
  - Chief note: "02:05 11/21: blobs unchanged; no eligible item; both no gap. JuniorHome tip by LLC bot." LLC skip path: `JuniorHome/scripts/advance_gguf_operator_sets_named_20261120.py` (not touched).
  - Roadmap blobs: `JuniorHome docs/ROADMAP.md` 4efb666, `JuniorLLM docs/BETA_TO_OS.md` bac6923.
- bench dir: `/workspace/bench0255_1121` (iq.py, patch.py, gen.py, run1.txt, run2.txt, xseed*.txt, results_*.json, tables.md)

## Script (`iq.py`)

```python
#!/usr/bin/env python3
"""Integer-code K scores (khv4hq), P-side V scales (khv4hqp) and 4 hot channels (khv4h4) for khv4h (02:55 11/21). Derived from kh.py (18:55 11/20).
khv4hq: K scores from int8 codes, one block-diag GEMM [Q per 64-ch group] @ codes^T, per-token group scales applied to the (G,2,T) partials (no dequant multiply).
khv4hqp: khv4hq plus V scales folded into P per channel group (P*s_g) @ codes_g, so V is only LUT-unpacked + cast.
khv4h4: khv4h with 4 hot K channels (subset of the top-6 pick).
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
FMTS = ("a8", "k8v4", "khv4h", "khv4hq", "khv4hqp", "khv4h4")
NPRE = 256
A8FAM = ("a8", "a8k32", "a8khot")
WB = {"u32": 16.65, "int8": 66.34}   # MiB, full bitnetd layer weights from 18:55 11/18 run1 (not re-timed)
ROUNDS, REPS = 7, 3
POOL = ThreadPoolExecutor(W)
SEED = int(os.environ.get("SEED", "11210255")); rng = np.random.default_rng(SEED)
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
BOLD["khv4hq"] = BOLD["khv4hqp"] = BOLD["khv4h"]; BOLD["khv4h4"] = (HD - 4) + 2 * 4 + 4 + 72
BWIN = {"a8": 2 * b8(64, 4), "k8v4": 2 * b8(64, 4), "k8v5": 2 * b8(64, 4), "k8v5h": 2 * b8(64, 2),
        "a8k32": BOLD["a8k32"], "a8khot": BOLD["a8khot"], "khv5h": (HD - NHOT) + 2 * NHOT + 4 + b8(64, 2)}
for _f in ("khv4h", "khv5hf", "khv5hp", "khv4hf", "khv4hq", "khv4hqp"): BWIN[_f] = BWIN["khv5h"]
BWIN["khv4h4"] = (HD - 4) + 2 * 4 + 4 + b8(64, 2)
HOTIDX = {"a8khot", "khv5h", "khv4h", "khv5hf", "khv5hp", "khv4hf", "khv4hq", "khv4hqp", "khv4h4"}
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
        s.hidx4 = s.hidx[:, :4]
        s.khot4 = np.stack([k[h][:, s.hidx4[h]] for h in range(KVH)]).astype(np.float16)
        kz4 = k.copy()
        for h in range(KVH): kz4[h][:, s.hidx4[h]] = 0
        s.kz48h, s.kzs48h = q8(kz4, 64, True)
    def bytes(s, fmt):
        wb = max(0, s.n - WIN); return KVH * (wb * BOLD[fmt] + (s.n - wb) * BWIN[fmt] + ((4 if fmt == "khv4h4" else NHOT) if fmt in HOTIDX else 0))
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
        if fmt in ("khv4hq", "khv4hqp"):
            C = s.kz8h[h, a:b].astype(np.float32); sc = s.kzs8h[h, a:b].astype(np.float32)
            Qe = np.zeros((2 * G, HD), np.float32); Qe[0::2, :64] = Q[:, :64]; Qe[1::2, 64:] = Q[:, 64:]
            R = (Qe @ C.T).reshape(G, 2, b - a); Sx = R[:, 0] * sc[:, 0]; Sx += R[:, 1] * sc[:, 1]
            Sx += Q[:, s.hidx[h]] @ s.khot[h, a:b].astype(np.float32).T; return Sx
        if fmt == "khv4h4":
            return Q @ dq8(s.kz48h[h, a:b], s.kzs48h[h, a:b]).T + Q[:, s.hidx4[h]] @ s.khot4[h, a:b].astype(np.float32).T
        if fmt in ("khv5h", "khv4h"):
            return Q @ dq8(s.kz8h[h, a:b], s.kzs8h[h, a:b]).T + Q[:, s.hidx[h]] @ s.khot[h, a:b].astype(np.float32).T
        if fmt == "k8v5h": return Q @ dq8(s.k8h[h, a:b], s.ks8h[h, a:b]).T
        return Q @ dq8(s.k8[h, a:b], s.ks8[h, a:b]).T
    def V(s, fmt, h, a, b, old):
        if old and fmt == "k8v4": return dq4(s.v4[h, a:b], s.v4s[h, a:b])
        if old and fmt == "k8v5": return dq5(s.v5lo[h, a:b], s.v5hb[h, a:b], s.v5s[h, a:b])
        if old and fmt in ("khv4h", "khv4hf", "khv4hq", "khv4h4"): return dq4(s.v4h[h, a:b], s.v4hs[h, a:b].astype(np.float32))
        if old and fmt in ("k8v5h", "khv5h", "khv5hf", "khv5hp"): return dq5(s.v5hlo[h, a:b], s.v5hhb[h, a:b], s.v5hs[h, a:b])
        if fmt in ("k8v5h", "khv5h", "khv4h", "khv5hf", "khv5hp", "khv4hf", "khv4hq", "khv4h4"): return dq8(s.v8h[h, a:b], s.vs8h[h, a:b])
        return dq8(s.v8[h, a:b], s.vs8[h, a:b])
    def PV(s, h, a, b, old):  # khv4hqp: V scales applied to P per channel group, V only unpacked + cast
        if old: C = np.take(LUT16, s.v4h[h, a:b]).view(np.int8).reshape(b - a, HD).astype(np.float32); sc = s.v4hs[h, a:b].astype(np.float32)
        else: C = s.v8h[h, a:b].astype(np.float32); sc = s.vs8h[h, a:b].astype(np.float32)
        ng = sc.shape[1]; gs = HD // ng
        def f(P):
            o = np.empty((P.shape[0], HD), np.float32)
            for g in range(ng): np.matmul(P * sc[:, g], C[:, g * gs:(g + 1) * gs], out=o[:, g * gs:(g + 1) * gs])
            return o
        return f
def online(m, l, acc, S, Vfn):
    mn = np.maximum(m, S.max(1, keepdims=True)); al = np.exp(m - mn); S -= mn; np.exp(S, out=S)
    l *= al; l += S.sum(1, keepdims=True); acc *= al; acc += Vfn(S); return mn
def tile(kv, fmt, h, a, b, Q, st, old):
    if fmt == "khv4hqp": return online(*st, kv.S(fmt, h, a, b, Q, old), kv.PV(h, a, b, old))
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
                for refn in ("a8", "k8v4", "khv4h"):
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

The KV format line has stopped paying off in numpy, so this rotates to the whole decode step. On the full bitnetd layer at B = 8 / ctx 8192 (also ctx 2048 and B = 1), with u32-LUT 2-bit weights and the khv4h4 cache:
- What share of each decode step's wall time goes to attention vs the 2-bit weight GEMVs (QKV, O, gate/up/down)?
- If attention is under ~25% of the step, the 0.85x attention speed of khv4h4 vs K8V4 costs < 4% end-to-end tok/s. Is the 1.80x memory win then the deciding factor for i2sd?
- At what context length does attention cross 50% of the step?

close: sha256(prev || this)[:16] = 98dc8a4f1898c9d4
