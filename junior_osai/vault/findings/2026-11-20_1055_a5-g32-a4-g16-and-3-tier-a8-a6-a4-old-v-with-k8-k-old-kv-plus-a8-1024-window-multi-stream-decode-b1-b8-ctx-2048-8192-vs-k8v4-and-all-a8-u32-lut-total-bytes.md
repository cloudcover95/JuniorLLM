---
local: true
date: 2026-11-20
wave: "2026-11-20 10:55 (ran ~10:48–11:00 MT; Chief last_run 2026-11-20T17:10:00Z = 10:10 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-stream, kv-cache, a5-v, a4-g16-v, a6-v, three-tier, recency, k8v4, a8-g64, recent-window, gqa, numpy, not-viable]
---

# Wider old V works and finer groups don't. With K8 K at B = 8 / ctx 8192, **A5 g32 old V** (nibble plane + 1-bit plane, +16 B/token) lifts min cos from **0.98489** (K8V4) to **0.98832** on this wave's seed, and from **0.99138** to **0.99642** on the replayed 05:55 data. That closes 62% / 82% of the K8V4 → all-A8 gap. It costs **1.051x** K8V4 total bytes (90.71 vs 86.34 MiB, just over the 1.05x bar) and runs at 0.83–0.85x of K8V4's attention speed. **A4 g16** V costs the same bytes but closes only 20–23%, at ~0.99x speed. The **3-tier** cache (A8 1024 → A6 2048 → A4 oldest, 1.029x bytes) does nothing at ctx 8192: −0.00115 / +0.00035 vs K8V4. Most old tokens stay A4, and they are now double-rounded A6 → A4. The ≥ 0.995 target is reached only on the 05:55 data (k8v5 0.99642). On this seed even all-A8 is 0.99044. Total memory vs int8 + A8 drops from K8V4's 1.75x to 1.67x (k8v5). No new 1.5x, and no improvement on K8V4 at equal bytes. Not viable. Keep K8V4. A5 V is the best accuracy option if ~5% more KV is acceptable.

## Question

[[2026-11-20_0555_hadamard-rotated-a4-g32-old-v-with-k8-kc6-kc5-k-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-and-all-a8-u32-lut-total-bytes]] found rotation can't fix plain 4-bit Gaussian rounding in V. Its next question, with K8 K at B = 8 / ctx 8192 (also ctx 2048, B = 1):
- Does (a) A5 g32 old V (+16 B/token), (b) A4 g16 old V (+16 B/token of scales), or (c) a 3-tier recency cache (A8 last 1024 → A6 V next 2048 → A4 g32 V oldest) lift min attention cos to ≥ 0.995 at ≤ 1.05x the K8V4 bytes?
- What does each cost in attention speed vs K8V4 and all-A8?

## Method

- Same attention-only microbench and synthetic K/V generator as 05:55 11/19 – 05:55 11/20 (`hv4.py` → `vw.py`). Gaussian K/V with a log-normal per-channel spread (K σ 0.5 plus 6 planted 6x hot channels per head, V σ 0.3), slow positional drift on K. New seed 11201055.
- Cross-check: `SEED=11200555` replays hv4.py's RNG stream (including its HR sign draw), so K/V/Q match the 05:55 11/20 data exactly. a8 and k8v4 reproduce 05:55's 0.99756 / 0.99138 bit-for-bit.
- Shapes: B ∈ {1, 8} streams × 5 KV heads × 4 q heads (GQA 20/5), HD 128, ctx ∈ {2048, 8192}. The last 1024 tokens stay at A8 g64 for K and V. Old K is always A8 g64 (K8).
- Old-V variants (bytes per old token per head, V only; K8 K adds 136 B):
  - `k8v4`: A4 g32, 80 B (current rec).
  - `k8v5`: A5 g32 (lo-nibble plane + packed 5th-bit plane + f32 scale per 32), 96 B.
  - `k8v4g16`: A4 g16, 96 B.
  - `k8t3`: 3-tier. The next 2048 tokens past the window are A6 g32 (112 B), the oldest are A4 g32 (80 B). The A4 tier is re-quantized from the A6-dequantized V, as a real A8 → A6 → A4 demotion would be. At ctx 2048 every old token is still in the A6 tier.
  - `k8v6`: A6 g32 for all old V, 112 B. Reference upper bound only, 1.10x bytes.
- Attention: online softmax per (stream, KV head) on a W = 8 thread pool, `OPENBLAS_NUM_THREADS=1`, TK ∈ {1024, 2048}, best TK per variant, 7 rounds × 3 reps, median (min–max). k8t3 runs its A4 and A6 segments as separate tiles.
- Side costs (one thread, 40 stream-heads at B = 8, 7 rounds × 200): write one aged-out V token per head as A4 / A5 / A4 g16 / A6, and the k8t3 demotion (dequant A6 + A4 quant).
- Total bytes = full bitnetd layer weights from 18:55 11/18 run1 (u32 LUT 16.65 MiB, int8 66.34 MiB; not re-timed) + KV measured at B = 8.
- Error: min over (stream, head) of cos(attention output, exact f32-KV softmax). The two runs gave identical error. Timing differs.

## Command

```
cd /workspace/bench1055_1120
OPENBLAS_NUM_THREADS=1 python3 vw.py results_run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 python3 vw.py results_run2.json > run2.txt
SEED=11200555 OPENBLAS_NUM_THREADS=1 python3 vw.py results_xseed0555.json > xseed0555.txt   # 05:55 data cross-check
```

## Results: attention ms per decode step, all streams (run1 / run2, best TK, median)

| ctx / B | a8 | k8v4 | k8v5 | k8v4g16 | k8t3 | k8v6 |
|---|---|---|---|---|---|---|
| 2048 / 1 | 0.94 / 1.17 | 1.30 / 1.45 | 1.41 / 1.49 | 1.35 / 1.43 | 1.71 / 1.59 | 1.64 / 1.60 |
| 2048 / 8 | 6.27 / 6.40 | 10.44 / 9.90 | 11.79 / 11.36 | 9.52 / 10.88 | 12.13 / 13.57 | 12.11 / 15.46 |
| 8192 / 1 | 3.95 / 3.94 | 4.10 / 5.28 | 4.63 / 4.39 | 4.34 / 4.37 | 5.18 / 4.66 | 5.66 / 5.44 |
| 8192 / 8 | 18.74 / 23.88 | 25.14 / 31.36 | 30.26 / 36.85 | 25.38 / 31.61 | 28.53 / 34.28 | 38.07 / 43.52 |

Spread at B = 8 / ctx 8192 (all TK 2048, ms, run1 | run2): a8 18.27–20.59 | 23.67–24.57; k8v4 24.91–28.96 | 30.97–33.86; k8v5 28.82–31.99 | 36.13–39.05; k8v4g16 24.56–25.68 | 30.53–32.16; k8t3 27.55–29.42 | 33.55–37.86; k8v6 35.64–41.56 | 42.57–45.95. Run2 is ~20% slower across the board (box load). At B = 8 the ratios below agree within ±0.07 between runs. At B = 1 they are noisier (up to ±0.3).

Speed ratio (best-TK median, run1 | run2), variant vs K8V4 and vs all-A8:

| ctx / B | k8v5 vs k8v4 | k8v4g16 vs k8v4 | k8t3 vs k8v4 | k8v6 vs k8v4 | k8v4 vs a8 | k8v5 vs a8 | k8v4g16 vs a8 | k8t3 vs a8 |
|---|---|---|---|---|---|---|---|---|
| 2048 / 1 | 0.93 / 0.97 | 0.96 / 1.02 | 0.76 / 0.91 | 0.80 / 0.91 | 0.72 / 0.80 | 0.67 / 0.78 | 0.70 / 0.82 | 0.55 / 0.73 |
| 2048 / 8 | 0.89 / 0.87 | 1.10 / 0.91 | 0.86 / 0.73 | 0.86 / 0.64 | 0.60 / 0.65 | 0.53 / 0.56 | 0.66 / 0.59 | 0.52 / 0.47 |
| 8192 / 1 | 0.88 / 1.20 | 0.94 / 1.21 | 0.79 / 1.13 | 0.72 / 0.97 | 0.96 / 0.75 | 0.85 / 0.90 | 0.91 / 0.90 | 0.76 / 0.84 |
| 8192 / 8 | 0.83 / 0.85 | 0.99 / 0.99 | 0.88 / 0.92 | 0.66 / 0.72 | 0.75 / 0.76 | 0.62 / 0.65 | 0.74 / 0.76 | 0.66 / 0.70 |

## Results: min cos vs exact f32 attention (identical in both runs)

Seed 11201055 (this wave):

| ctx / B | a8 | k8v4 | k8v5 | k8v4g16 | k8t3 | k8v6 |
|---|---|---|---|---|---|---|
| 2048 / 1 | 0.99876 | 0.99476 | 0.99769 | 0.99562 | 0.99855 | 0.99855 |
| 2048 / 8 | 0.99450 | 0.99243 | 0.99433 | 0.99383 | 0.99443 | 0.99443 |
| 8192 / 1 | 0.99044 | 0.98489 | 0.98832 | 0.98618 | 0.98374 | 0.98995 |
| 8192 / 8 | 0.99044 | 0.98489 | 0.98832 | 0.98618 | 0.98374 | 0.98995 |

Seed 11200555 replay (05:55 11/20 data; a8 / k8v4 match the 05:55 note exactly):

| ctx / B | a8 | k8v4 | k8v5 | k8v4g16 | k8t3 | k8v6 |
|---|---|---|---|---|---|---|
| 2048 / 1 | 0.99668 | 0.99449 | 0.99643 | 0.99522 | 0.99672 | 0.99672 |
| 2048 / 8 | 0.99668 | 0.99177 | 0.99643 | 0.99445 | 0.99672 | 0.99672 |
| 8192 / 1 | 0.99776 | 0.99138 | 0.99666 | 0.99260 | 0.99188 | 0.99753 |
| 8192 / 8 | 0.99756 | 0.99138 | 0.99642 | 0.99260 | 0.99173 | 0.99741 |

Share of the K8V4 → all-A8 min-cos gap closed at B = 8 / ctx 8192 (seed 11201055 | replay): k8v5 62% | 82%; k8v4g16 23% | 20%; k8t3 −21% | 6%; k8v6 91% | 98%.

Old-token V relative Frobenius error vs f32 V (median over stream-heads, seed 11201055, ctx 8192): A4 g32 0.1101; A5 g32 0.0517; A4 g16 0.0936; A6 g32 0.0257; 3-tier overall 0.0968, of which the A4 tier is 0.1133 (A6 → A4 double rounding, vs 0.1101 direct).

## Results: bytes at B = 8

| variant | KV MiB ctx 2048 | KV MiB ctx 8192 | total ctx 8192 (u32 + KV) | vs int8 + A8 (151.34 MiB) | vs int8 + same KV | total / u32 + K8V4 |
|---|---|---|---|---|---|---|
| a8 | 21.25 | 85.00 | 101.65 | 1.49x | 1.49x | 1.177x |
| k8v4 | 19.06 | 69.69 | 86.34 | 1.75x | 1.58x | 1.000x |
| k8v5 | 19.69 | 74.06 | 90.71 | 1.67x | 1.55x | 1.051x |
| k8v4g16 | 19.69 | 74.06 | 90.71 | 1.67x | 1.55x | 1.051x |
| k8t3 | 20.31 | 72.19 | 88.84 | 1.70x | 1.56x | 1.029x |
| k8v6 | 20.31 | 78.44 | 95.09 | 1.59x | 1.52x | 1.101x |

("vs int8 + A8" = int8 weights 66.34 MiB + all-A8 KV 85.00 MiB = 151.34 MiB, divided by the row's total. At ctx 2048 / B = 8 the totals are a8 37.90, k8v4 35.71, k8v5 / k8v4g16 36.34, k8t3 / k8v6 36.96 MiB, i.e. 2.31x / 2.45x / 2.41x / 2.37x vs int8 + A8.)

Side costs, B = 8 / 40 stream-heads, µs per decode step (run1 | run2): write A4 g32 29.93 | 33.70; A5 g32 33.48 | 37.27; A4 g16 45.75 | 47.65; A6 g32 40.77 | 43.43; k8t3 demotion (dq A6 + q A4) 50.32 | 54.65 on top of the A6 write. Against a 25–37 ms attention step, all of these are ≤ 0.4%.

## Verdict

**not-viable.** No layout gives a new ≥ 1.5x, and none beats K8V4 at equal bytes:
- **k8v5 (A5 g32 V)** is a real accuracy lever. It cuts old-V error in half (0.110 → 0.052) and closes 62% / 82% of the gap to all-A8. It reaches ≥ 0.995 only on the 05:55 data (0.99642 vs 0.99138). It costs +5.1% total bytes (1.051x, just over the ≤ 1.05x bar), 1.67x vs int8 + A8 instead of 1.75x, and 0.83–0.89x of K8V4's attention speed at B = 8.
- **k8v4g16** buys little for the same +16 B (old-V error 0.110 → 0.094, 20–23% of the gap) at ~0.99x K8V4 speed. Spending the 16 B on a 5th bit beats spending it on scales by ~3–4x in gap closed.
- **k8t3** fits the byte bar (1.029x) but its A6 tier covers only 2048 of 7168 old tokens at ctx 8192. Its A4 tier is slightly worse than direct A4 (double rounding), so min cos is −0.00115 / +0.00035 vs K8V4, at 0.88–0.92x speed. Recency tiers only help when the worst head's attention mass is recent. Here it isn't.
- **≥ 0.995 at seed 11201055 is not reachable by any V change**: all-A8 is 0.99044 at ctx 8192. That floor comes from A8 g64 on hot-channel K/V and from softmax sharpness, not from the old-V width.

K8V4 stays the recommended old-token cache at 1.75x total below int8 + A8 (recorded 05:55 11/19, not new). If a deployment wants accuracy over the last 5% of KV bytes, k8v5 is the option to use, not g16 or the 3-tier cache.

Caveat: synthetic Gaussian V and two seeds. Real V caches with heavier tails may favour g16 more than this.

## Links

- previous: [[2026-11-20_0555_hadamard-rotated-a4-g32-old-v-with-k8-kc6-kc5-k-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-and-all-a8-u32-lut-total-bytes]]
- related: [[2026-11-20_0255_per-channel-a5-a6-k-old-kv-plus-a4-v-and-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-k4v4-and-all-a8-u32-lut-total-bytes]] (A5/A6 on K; same packers reused for V)
- this wave's pair files: none.
  - Quant Chief 10:05 assignment was pair_A no gap and pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-20T17:10:00Z).
  - Chief note: "10:05 11/20: blobs unchanged; i2sd userspace landed (JuniorHome d41804d); both no gap." LLC skip path: `JuniorHome/scripts/advance_gguf_operator_sets_named_20261120.py` (not touched).
  - Roadmap blobs: `JuniorHome docs/ROADMAP.md` 4efb666, `JuniorLLM docs/BETA_TO_OS.md` bac6923.
- bench dir: `/workspace/bench1055_1120` (vw.py, run1.txt, run2.txt, xseed0555.txt, results_run1.json, results_run2.json, results_xseed0555.json)

## Script (`vw.py`)

```python
#!/usr/bin/env python3
"""Wider / finer old-token V with K8 K, and a 3-tier recency cache (10:55 11/20). Derived from hv4.py (05:55 11/20).
Old-token K is always A8 g64 (K8). Last WIN=1024 tokens stay A8 g64 for K and V. Old-token V variants:
  k8v4    A4 g32 (nibble plane + f32 scale per 32) = 80 B/token/head      (current rec, K8V4)
  k8v5    A5 g32 (nibble plane + 1-bit plane + f32 scale per 32) = 96 B   (+16 B)
  k8v4g16 A4 g16 (nibble plane + f32 scale per 16) = 96 B                 (+16 B)
  k8t3    3-tier: A8 last 1024 -> A6 g32 V (112 B) next T2=2048 -> A4 g32 V oldest (80 B). The A4 tier is
          re-quantized from the A6-dequantized V (as a real demotion A8 -> A6 -> A4 would be).
  k8v6    A6 g32 for all old V (112 B), reference upper bound only.
Usage: OPENBLAS_NUM_THREADS=1 python3 vw.py out.json   (SEED=11200555 replays the 05:55 11/20 K/V/Q data)"""
import json, sys, time, statistics
from concurrent.futures import ThreadPoolExecutor
import numpy as np
KVH, G, HD, WIN, W, T2 = 5, 4, 128, 1024, 8, 2048; SC = np.float32(1 / np.sqrt(HD)); NG = HD // 32
CTXS, BS, TKS = (2048, 8192), (1, 8), (1024, 2048)
FMTS = ("a8", "k8v4", "k8v5", "k8v4g16", "k8t3", "k8v6")
NEW = ("k8v5", "k8v4g16", "k8t3", "k8v6")
VB = {"v4": HD // 2 + NG * 4, "v5": HD // 2 + HD // 8 + NG * 4, "v4g16": HD // 2 + (HD // 16) * 4, "v6": 3 * HD // 4 + NG * 4}
WB = {"u32": 16.65, "int8": 66.34}   # MiB, full bitnetd layer weights from 18:55 11/18 run1 (not re-timed)
ROUNDS, REPS = 7, 3
POOL = ThreadPoolExecutor(W)
import os
SEED = int(os.environ.get("SEED", "11201055")); rng = np.random.default_rng(SEED)
if SEED == 11200555: rng.choice([-1.0, 1.0], HD)   # replay hv4.py's HR sign draw so K/V/Q match the 05:55 11/20 data exactly
DEC = np.array([[(c & 15) - 8, (c >> 4) - 8] for c in range(256)], np.int8); LUT16 = DEC.view(np.uint16).ravel()
DECU = np.array([[c & 15, c >> 4] for c in range(256)], np.uint8); LUTU16 = DECU.view(np.uint16).ravel()
def q8(x, gs=64):
    g = x.reshape(*x.shape[:-1], HD // gs, gs); m = np.abs(g).max(-1) / 127.0 + 1e-12
    return np.clip(np.rint(g / m[..., None]), -127, 127).astype(np.int8).reshape(x.shape), m.astype(np.float32)
def dq8(c, s): return (c.reshape(*c.shape[:-1], HD // 64, 64).astype(np.float32) * s[..., None]).reshape(c.shape)
def pack4(c):
    n = c.shape[-1]; c = c.astype(np.uint8).reshape(*c.shape[:-1], n // 2, 2); return (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8)
def qsym(x, lim, gs):   # per-token symmetric group quant, codes in [-lim, lim]
    g = x.reshape(*x.shape[:-1], HD // gs, gs); m = np.abs(g).max(-1) / float(lim) + 1e-12
    return np.clip(np.rint(g / m[..., None]), -lim, lim).astype(np.int16).reshape(x.shape), m.astype(np.float32)
def q4(x, gs=32):
    c, m = qsym(x, 7, gs); return pack4(c + 8), m
def dq4(p, s, gs=32):
    u = np.take(LUT16, p).view(np.int8).reshape(*p.shape[:-1], HD // gs, gs)
    return (u.astype(np.float32) * s[..., None]).reshape(*p.shape[:-1], HD)
def q5(x):
    c, m = qsym(x, 15, 32); c = (c + 16).astype(np.uint8)
    return pack4(c & 15), np.packbits((c >> 4).astype(np.uint8), axis=-1, bitorder="little"), m
def unpack5(lo, hb):
    u = np.take(LUTU16, lo).view(np.uint8).reshape(lo.shape[0], HD)
    u |= np.unpackbits(hb, axis=-1, bitorder="little") << 4; u -= 16; return u.view(np.int8)
def dq5(lo, hb, s): return (unpack5(lo, hb).reshape(lo.shape[0], NG, 32).astype(np.float32) * s[..., None]).reshape(lo.shape[0], HD)
def pack6(c):
    c = c.astype(np.uint8).reshape(*c.shape[:-1], HD // 4, 4)
    b0 = c[..., 0] | (c[..., 1] << 6); b1 = (c[..., 1] >> 2) | (c[..., 2] << 4); b2 = (c[..., 2] >> 4) | (c[..., 3] << 2)
    return np.stack([b0, b1, b2], -1).astype(np.uint8).reshape(*c.shape[:-2], 3 * HD // 4)
def unpack6(p):
    b = p.reshape(p.shape[0], HD // 4, 3); b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    o = np.empty((p.shape[0], HD // 4, 4), np.uint8)
    np.bitwise_and(b0, 63, out=o[..., 0]); o[..., 1] = (b0 >> 6) | ((b1 & 15) << 2)
    o[..., 2] = (b1 >> 4) | ((b2 & 3) << 4); np.right_shift(b2, 2, out=o[..., 3]); o -= 32
    return o.view(np.int8).reshape(p.shape[0], HD)
def q6(x): c, m = qsym(x, 31, 32); return pack6(c + 32), m
def dq6(p, s): return (unpack6(p).reshape(p.shape[0], NG, 32).astype(np.float32) * s[..., None]).reshape(p.shape[0], HD)
def kvhist(n):  # same generator as 05:55 11/19 - 05:55 11/20
    ch = np.exp(rng.normal(0, 0.5, (KVH, 1, HD))).astype(np.float32); ch[:, :, rng.choice(HD, 6, replace=False)] *= 6
    k = (rng.standard_normal((KVH, n, HD)).astype(np.float32) + 0.3 * np.sin(np.arange(n) / 300.0)[None, :, None]) * ch
    v = rng.standard_normal((KVH, n, HD)).astype(np.float32) * np.exp(rng.normal(0, 0.3, (KVH, 1, HD))).astype(np.float32)
    return k.astype(np.float32), v.astype(np.float32)
def tiers(n):  # (a6_start, window_start): old tokens [0, mid) A4, [mid, wb) A6 for k8t3
    wb = max(0, n - WIN); return max(0, wb - T2), wb
class KV:
    def __init__(s, k, v):
        s.n = k.shape[1]; s.kf, s.vf = k, v
        (s.k8, s.ks8), (s.v8, s.vs8) = q8(k), q8(v)
        vd = dq8(s.v8, s.vs8)
        s.v4, s.v4s = q4(vd); s.v4g, s.v4gs = q4(vd, 16); s.v6, s.v6s = q6(vd)
        s.v5lo, s.v5hb, s.v5s = q5(vd)
        vd6 = np.stack([dq6(s.v6[h], s.v6s[h]) for h in range(KVH)]); s.v4t, s.v4ts = q4(vd6)   # A8 -> A6 -> A4 demotion
    def bytes(s, fmt):
        b8 = HD + HD // 64 * 4
        if fmt == "a8": return 2 * KVH * s.n * b8
        mid, wb = tiers(s.n); n8 = s.n - wb
        if fmt == "k8t3": vold = mid * VB["v4"] + (wb - mid) * VB["v6"]
        else: vold = wb * VB[fmt[2:]]
        return KVH * (2 * n8 * b8 + wb * b8 + vold)
def online(m, l, acc, S, Vfn):
    mn = np.maximum(m, S.max(1, keepdims=True)); al = np.exp(m - mn); S -= mn; np.exp(S, out=S)
    l *= al; l += S.sum(1, keepdims=True); acc *= al; acc += Vfn(S); return mn
def vdq(kv, h, a, b, fmt, tier):
    if fmt == "k8v4": return dq4(kv.v4[h, a:b], kv.v4s[h, a:b])
    if fmt == "k8v5": return dq5(kv.v5lo[h, a:b], kv.v5hb[h, a:b], kv.v5s[h, a:b])
    if fmt == "k8v4g16": return dq4(kv.v4g[h, a:b], kv.v4gs[h, a:b], 16)
    if fmt == "k8v6" or tier == 6: return dq6(kv.v6[h, a:b], kv.v6s[h, a:b])
    return dq4(kv.v4t[h, a:b], kv.v4ts[h, a:b])   # k8t3 oldest tier
def tile_a8(kv, h, a, b, Q, st):
    K = dq8(kv.k8[h, a:b], kv.ks8[h, a:b]); V = dq8(kv.v8[h, a:b], kv.vs8[h, a:b])
    return online(*st, Q @ K.T, lambda P: P @ V)
def tile_old(kv, h, a, b, Q, st, fmt, tier):
    K = dq8(kv.k8[h, a:b], kv.ks8[h, a:b]); V = vdq(kv, h, a, b, fmt, tier); return online(*st, Q @ K.T, lambda P: P @ V)
def head(kv, Q, h, fmt, TK):
    st = [np.full((G, 1), -np.inf, np.float32), np.zeros((G, 1), np.float32), np.zeros((G, HD), np.float32)]
    if fmt == "a8":
        for a in range(0, kv.n, TK): st[0] = tile_a8(kv, h, a, min(a + TK, kv.n), Q, st)
        return st[2] / st[1]
    mid, wb = tiers(kv.n)
    segs = ((0, mid, 4), (mid, wb, 6)) if fmt == "k8t3" else ((0, wb, 0),)
    for lo, hi, tier in segs:
        for a in range(lo, hi, TK): st[0] = tile_old(kv, h, a, min(a + TK, hi), Q, st, fmt, tier)
    for a in range(wb, kv.n, WIN): st[0] = tile_a8(kv, h, a, min(a + WIN, kv.n), Q, st)
    return st[2] / st[1]
def attn(kvs, q, fmt, TK):
    futs = [POOL.submit(head, kvs[s], q[s, h], h, fmt, TK) for s in range(len(kvs)) for h in range(KVH)]
    return np.stack([f.result() for f in futs])
def ref(kv, Q, h):
    S = Q @ kv.kf[h].T; S -= S.max(1, keepdims=True); P = np.exp(S); P /= P.sum(1, keepdims=True); return P @ kv.vf[h]
def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
def tmed(fn, inner):
    ts = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for _ in range(inner): fn()
        ts.append((time.perf_counter() - t0) / inner * 1e6)
    return dict(med=statistics.median(ts), lo=min(ts), hi=max(ts))
def main():
    out = {"rows": [], "compare": {}, "side": {}}
    print(f"{'cfg':30s} {'attn ms med (min-max)':>24s} {'KV MiB':>8s} {'min cos':>10s}")
    for n in CTXS:
        kvs8 = [KV(*kvhist(n)) for _ in range(max(BS))]
        q8all = (rng.standard_normal((max(BS), KVH, G, HD)).astype(np.float32) * SC)
        mid, wb = tiers(n)
        dqs = (("v4", lambda kv, h: dq4(kv.v4[h, :wb], kv.v4s[h, :wb])),
               ("v5", lambda kv, h: dq5(kv.v5lo[h, :wb], kv.v5hb[h, :wb], kv.v5s[h, :wb])),
               ("v4g16", lambda kv, h: dq4(kv.v4g[h, :wb], kv.v4gs[h, :wb], 16)),
               ("v6", lambda kv, h: dq6(kv.v6[h, :wb], kv.v6s[h, :wb])),
               ("t3", lambda kv, h: np.concatenate([dq4(kv.v4t[h, :mid], kv.v4ts[h, :mid]), dq6(kv.v6[h, mid:wb], kv.v6s[h, mid:wb])])))
        for nm, fn in dqs:
            e = [np.linalg.norm(fn(kv, h) - kv.vf[h, :wb]) / np.linalg.norm(kv.vf[h, :wb]) for kv in kvs8 for h in range(KVH)]
            out["compare"][f"Verr_{nm}_ctx{n}"] = float(np.median(e)); print(f"  old-V rel err {nm} ctx{n}: median {np.median(e):.5f} max {max(e):.5f}")
        if mid > 0:
            e = [np.linalg.norm(dq4(kv.v4t[h, :mid], kv.v4ts[h, :mid]) - kv.vf[h, :mid]) / np.linalg.norm(kv.vf[h, :mid]) for kv in kvs8 for h in range(KVH)]
            out["compare"][f"Verr_t3_a4tier_ctx{n}"] = float(np.median(e)); print(f"  old-V rel err t3 A4 tier (A6->A4) ctx{n}: median {np.median(e):.5f}")
        for B in BS:
            kvs, q = kvs8[:B], q8all[:B]
            R = np.stack([np.stack([ref(kvs[s], q[s, h], h) for h in range(KVH)]) for s in range(B)]).reshape(B * KVH, G, HD)
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
                    c = min(cos(o[i], R[i]) for i in range(B * KVH))
                    key = f"{fmt}_TK{TK}_ctx{n}_B{B}"
                    r = dict(key=key, med=statistics.median(ts), lo=min(ts), hi=max(ts), ts=ts, mib=B * kvs[0].bytes(fmt) / 2**20, cos=c)
                    res[(fmt, TK)] = r; out["rows"].append(r)
                    print(f"{key:30s} {r['med']:8.2f} ({r['lo']:.2f}-{r['hi']:.2f})      {r['mib']:8.2f} {c:10.7f}")
            best = {f: min((res[(f, t)] for t in TKS), key=lambda r: r["med"]) for f in FMTS}
            for f in NEW:
                for refn in ("a8", "k8v4"):
                    k = f"{f}_vs_{refn}_ctx{n}_B{B}"; r = best[refn]["med"] / best[f]["med"]
                    out["compare"][k] = dict(speed=r, bytes=best[refn]["mib"] / best[f]["mib"], dcos=best[f]["cos"] - best[refn]["cos"])
                    print(f"  {k:28s} speed {r:5.2f}x  KV bytes {best[refn]['mib'] / best[f]['mib']:5.2f}x  dcos {best[f]['cos'] - best[refn]['cos']:+.5f}")
            if B == 8:
                for wf in ("u32", "int8"):
                    for f in FMTS: out["compare"][f"total_{wf}_{f}_ctx{n}_B8"] = WB[wf] + best[f]["mib"]
                t = lambda wf, f: out["compare"][f"total_{wf}_{f}_ctx{n}_B8"]
                for f in FMTS:
                    print(f"  total ctx{n} B8 u32+{f}: {t('u32', f):7.2f} MiB | vs int8+a8 {t('int8','a8')/t('u32', f):5.2f}x | vs int8+same {t('int8', f)/t('u32', f):5.2f}x | u32+{f} / u32+k8v4 {t('u32', f)/t('u32','k8v4'):5.3f}x")
    # side costs at B = 8 (40 stream-heads), single thread, microseconds per decode step (one aged-out token per head)
    X = rng.standard_normal((8 * KVH, HD)).astype(np.float32)
    p6, s6 = q6(X)
    out["side"]["write_q4"] = tmed(lambda: q4(X), 200)
    out["side"]["write_q5"] = tmed(lambda: q5(X), 200)
    out["side"]["write_q4g16"] = tmed(lambda: q4(X, 16), 200)
    out["side"]["write_q6"] = tmed(lambda: q6(X), 200)
    out["side"]["t3_demote_dq6_q4"] = tmed(lambda: q4(dq6(p6, s6)), 200)
    for k, v in out["side"].items(): print(f"  side {k:18s} B8 (40 stream-heads): {v['med']:8.2f} us ({v['lo']:.2f}-{v['hi']:.2f})")
    json.dump(out, open(sys.argv[1], "w"), indent=1)
main()
```

## Next question

A5 V buys the accuracy but overshoots the byte bar by 0.1%. Can it be funded elsewhere? At B = 8 / ctx 8192, does K8 + A5 g32 V with f16 instead of f32 group scales (on old K, old V and the A8 window) land within ~1.01x of K8V4 total bytes (220 vs 216 B per old token, 264 vs 272 B per window token → ~1.011x KV, ~1.009x total) with min cos within 0.0005 of f32-scale k8v5 (0.98832 / 0.99642)? And at seed 11201055, where even all-A8 stops at 0.99044, does A8 g32 K (or the 6 hot K channels kept in f16) lift that floor, and at what attention speed?

close: sha256(prev || this)[:16] = 67b2aeb2e8a14dd4
