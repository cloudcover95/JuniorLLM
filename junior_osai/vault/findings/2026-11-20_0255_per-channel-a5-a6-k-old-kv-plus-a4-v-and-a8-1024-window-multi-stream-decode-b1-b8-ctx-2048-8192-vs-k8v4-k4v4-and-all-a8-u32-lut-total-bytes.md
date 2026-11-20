---
local: true
date: 2026-11-20
wave: "2026-11-20 02:55 (ran ~02:48–02:55 MT; Chief last_run 2026-11-20T09:15:00Z = 02:15 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-stream, kv-cache, per-channel, a5, a6, bit-packing, k8v4, a4, a8-g64, recent-window, gqa, error-split, numpy, not-viable]
---

# A middle K width does not beat K8V4. At B = 8 / ctx 8192, per-channel **A6** K + A4 V reaches min cos **0.99002**, within the bar (K8V4 0.99142 − 0.002). But it saves only **1.08x** total bytes vs K8V4 (79.78 vs 86.34 MiB) and runs attention at **0.60x** of K8V4. Per-channel **A5** K saves **1.15x** (75.40 MiB) but drops to **0.98796** (−0.0035). The error split shows why: K8V4's whole gap to all-A8 (0.0058) comes from V at A4. K at A6 costs only 0.0007. Not viable. Keep K8V4. The next fix belongs on the V side.

## Question

[[2026-11-19_1855_top-k-hot-k-channels-a8-plus-per-channel-a4-rest-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-kivi-k4v4-k8v4-and-all-a8-u32-lut-total-bytes]] found that three A4 K fixes (rotation, per-channel, top-k hot channels) all stall below K8V4, so the limit is the 4-bit K width. Its next question:
- Does per-channel A6 K (4 codes in 3 bytes) or A5 K (nibble plane + 1-bit plane), each with an f32 scale per (32-token block, channel) and with V at A4 g32, reach min cos ≥ K8V4 − 0.002 at ≥ 1.1x fewer total bytes than K8V4 at B = 8 / ctx 8192?
- What does the 5/6-bit unpack cost per decode step in numpy?
- How does K8V4's error split into a K-only part and a V-only part?

## Method

- Same attention-only microbench and synthetic K/V generator as 05:55–18:55 11/19 (`mixk.py` → `a6k.py`). Gaussian K/V with a log-normal per-channel spread, 6 planted hot K channels × 6 per head and a slow positional drift. New seed 11200255.
- Shapes: B ∈ {1, 8} streams × 5 KV heads × 4 q heads (GQA 20/5), HD 128, ctx ∈ {2048, 8192}. The last 1024 tokens stay at A8 g64.
- Attention: online softmax per (stream, KV head) on a W = 8 thread pool, `OPENBLAS_NUM_THREADS=1`, TK ∈ {1024, 2048}, best TK per variant.
- Old-token K is quantized from the A8-dequantized K, symmetric per channel, with an f32 scale per (32-token block, channel):
  - `kc4`: [−7, 7], nibble-packed, 64 B/token
  - `kc5`: [−15, 15], low-nibble plane + 1-bit high plane (`np.packbits`, little bit order), 80 B/token
  - `kc6`: [−31, 31], 4 codes in 3 bytes with shift/mask, 96 B/token
  - each also carries 16 B/token of scales.
- Decode unpacks with a uint16 nibble LUT (kc4/kc5), `np.unpackbits` (kc5) or 4 shift/mask ops (kc6), then int8 → f32 × scale. Pack/unpack round-trip is checked exact for kc5 and kc6.
- Variants: `a8` (all A8 g64), `k8v4` (05:55 11/19 recommendation), `kc4v4`, `kc5v4`, `kc6v4`. The error split uses `kc5v8` / `kc6v8` (old V kept at A8 g64, so K is the only extra error) together with `k8v4` vs `a8` (V is the only extra error).
- Side costs:
  - old-K relative Frobenius error vs f32 K
  - isolated K unpack + dequant over all old tokens of one B = 8 / ctx 8192 step (7168 tokens × 40 heads, one thread, 2048-token tiles)
  - each measured over 7 rounds.
- Total bytes = full bitnetd layer weights from 18:55 11/18 run1 (u32 LUT 16.65 MiB, int8 66.34 MiB; not re-timed) + KV measured at B = 8.
- Error: min over (stream, head) of cos(attention output, exact f32-KV softmax). Each config ran 7 rounds × 3 reps, reported as median (min–max). Two independent runs gave identical error; timing differs.

## Command

```
cd /workspace/bench0255_1120
OPENBLAS_NUM_THREADS=1 python3 a6k.py results_run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 python3 a6k.py results_run2.json > run2.txt
```

## Results: attention ms per decode step, all streams (run1 / run2, best TK, median)

| ctx / B | a8 | k8v4 | kc4v4 | kc5v4 | kc6v4 | kc5v8 | kc6v8 | KV MiB a8 / k8v4 / kc4v4 / kc5v4 / kc6v4 / kc5v8 / kc6v8 |
|---|---|---|---|---|---|---|---|---|
| 2048 / 1 | 1.06 / 0.95 | 1.39 / 1.16 | 1.29 / 1.20 | 1.47 / 1.43 | 1.75 / 1.63 | 1.50 / 1.24 | 1.51 / 1.54 | 2.66 / 2.38 / 2.11 / 2.19 / 2.27 / 2.46 / 2.54 |
| 2048 / 8 | 5.97 / 5.76 | 10.33 / 8.82 | 9.52 / 9.42 | 10.92 / 10.78 | 12.36 / 13.20 | 10.15 / 10.07 | 11.59 / 11.51 | 21.25 / 19.06 / 16.88 / 17.50 / 18.12 / 19.69 / 20.31 |
| 8192 / 1 | 3.04 / 3.20 | 4.01 / 3.83 | 4.18 / 4.25 | 5.83 / 4.82 | 5.47 / 5.34 | 5.17 / 4.38 | 5.23 / 5.27 | 10.62 / 8.71 / 6.80 / 7.34 / 7.89 / 9.26 / 9.80 |
| 8192 / 8 | 19.46 / 19.02 | 24.58 / 24.75 | 27.74 / 27.55 | 32.66 / 33.08 | 41.12 / 41.26 | 30.25 / 32.17 | 35.54 / 41.36 | 85.00 / 69.69 / 54.38 / 58.75 / 63.12 / 74.06 / 78.44 |

Spread at B = 8 / ctx 8192 (best TK, run1 | run2, ms):

| variant | run1 | run2 |
|---|---|---|
| a8 | 18.66–19.98 | 18.59–20.05 |
| k8v4 | 23.45–26.96 | 24.06–25.65 |
| kc4v4 | 26.83–29.55 | 26.68–28.49 |
| kc5v4 | 31.86–36.50 | 31.67–33.56 |
| kc6v4 | 38.56–46.24 | 39.22–44.12 |
| kc5v8 | 29.03–31.39 | 30.82–34.10 |
| kc6v8 | 35.29–37.09 | 40.70–42.63 |

Speed vs k8v4 at B = 8 / ctx 8192: kc5v4 0.75x, kc6v4 0.60x, kc4v4 0.89x.

Error (min attention-output cos; identical in both runs):

| ctx / B | a8 | k8v4 | kc4v4 | kc5v4 | kc6v4 | kc5v8 | kc6v8 |
|---|---|---|---|---|---|---|---|
| 2048 / 1 | 0.99778 | 0.99456 | 0.95995 | 0.98832 | 0.99384 | 0.99200 | 0.99711 |
| 2048 / 8 | 0.99525 | 0.99131 | 0.95995 | 0.98832 | 0.98965 | 0.99200 | 0.99499 |
| 8192 / 1 | 0.99837 | 0.99306 | 0.96569 | 0.99103 | 0.99222 | 0.99666 | 0.99744 |
| 8192 / 8 | 0.99722 | 0.99142 | 0.95651 | 0.98796 | 0.99002 | 0.99255 | 0.99657 |

Error split at B = 8 / ctx 8192 (loss in min cos vs all-A8 0.99722):

| part | loss |
|---|---|
| V at A4, K at A8 (k8v4) | 0.00580 |
| K at A6 only (kc6v8) | 0.00065 |
| K at A5 only (kc5v8) | 0.00467 |
| K at A6 + V at A4 (kc6v4) | 0.00720 |
| K at A5 + V at A4 (kc5v4) | 0.00926 |

Old-K relative error vs f32 K (median over heads, ctx 8192): A8 g64 0.0127, kc6 0.0253, kc5 0.0470, kc4 0.0977. Each bit removed roughly doubles the error.

Unpack cost: isolated K unpack + dequant at B = 8 / ctx 8192, one thread (run1 / run2 median, ms):

| K format | run1 | run2 |
|---|---|---|
| A8 g64 | 16.78 | 16.97 |
| kc4 | 28.00 | 29.19 |
| kc5 | 44.66 | 55.58 |
| kc6 | 58.28 | 59.49 |

kc6 is about 3.5x the A8 dequant and 2x kc4. That is roughly 5.2–5.3 ms of the 8-thread step, which accounts for most of the 0.60x attention speed.

Total bytes per layer at B = 8 (weights + KV, MiB):

| ctx | u32 + a8 | u32 + K8V4 | u32 + kc4v4 | u32 + kc5v4 | u32 + kc6v4 | int8 + a8 | kc6v4 vs int8+a8 | kc6v4 vs u32+K8V4 | kc5v4 vs u32+K8V4 |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 37.90 | 35.71 | 33.52 | 34.15 | 34.77 | 87.59 | 2.52x | 1.03x | 1.05x |
| 8192 | 101.65 | 86.34 | 71.03 | 75.40 | 79.78 | 151.34 | 1.90x | 1.08x | 1.15x |

## Verdict

**not-viable.** No middle K width passes both bars:
- A6 K holds accuracy (0.99002, 0.0014 below K8V4 and inside the 0.002 bar) but saves only 1.08x total bytes vs K8V4, below the 1.1x bar. It also costs 0.60x attention speed because the 3-byte shift/mask unpack runs about 3.5x slower than the A8 dequant.
- A5 K saves 1.15x total bytes but misses accuracy (0.98796, 0.0035 below K8V4).

The error split moves the problem. K8V4's entire gap to all-A8 is the A4 V cache (0.0058), while A6 K adds only 0.0007. So the next accuracy gain is on V, not K. K8V4 stays the recommended old-token cache. The 2-bit ternary weights still carry the real memory win: K8V4 total is 1.75x below int8 + A8 at B = 8 / ctx 8192.

Caveat: synthetic K/V (fixed hot channels, Gaussian V), numpy unpack. A fused C/Metal 6-bit unpack would change the speed side but not the byte ratio.

## Links

- previous: [[2026-11-19_1855_top-k-hot-k-channels-a8-plus-per-channel-a4-rest-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-kivi-k4v4-k8v4-and-all-a8-u32-lut-total-bytes]]
- related: [[2026-11-19_1455_kivi-per-channel-a4-k-sym-and-asym-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k4v4-k8v4-and-all-a8-u32-lut-total-bytes]], [[2026-11-19_0555_k8v4-a8-g64-keys-all-tokens-a4-g32-old-values-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k4v4-and-all-a8-u32-lut-total-bytes]]
- this wave's pair files: none.
  - Quant Chief 02:05 assignment was pair_A no gap and pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-20T09:15:00Z).
  - Chief note: "02:05 11/20: blobs unchanged; both no gap. JuniorHome 8afb854 re-advances landed GGUF operator, left alone."
  - Roadmap blobs: `JuniorHome docs/ROADMAP.md` 4efb666, `JuniorLLM docs/BETA_TO_OS.md` bac6923.
- bench dir: `/workspace/bench0255_1120` (a6k.py, run1.txt, run2.txt, results_run1.json, results_run2.json)

## Script (`a6k.py`)

```python
#!/usr/bin/env python3
"""Middle K width for old tokens (02:55 11/20). Derived from mixk.py (18:55 11/19) / kivi4.py (14:55 11/19).
Old-token K (aged out of the A8 g64 window, quantized from the A8-dequantized K) uses symmetric per-channel codes with
an f32 scale per (32-token block, channel):
  kc4  A4 [-7,7]   nibble-packed            (HD/2 B per token)
  kc5  A5 [-15,15] nibble plane + 1-bit plane (np.packbits)  (HD/2 + HD/8 B)
  kc6  A6 [-31,31] 4 codes in 3 bytes (bit ops)              (3*HD/4 B)
Old-token V: per-token A4 g32 (v4) or A8 g64 (v8). Last WIN=1024 tokens stay A8 g64 for K and V.
Variants: a8 (all A8), k8v4 (05:55 11/19 rec), kc4v4, kc5v4, kc6v4, plus kc6v8 / kc5v8 (K-only error split).
Usage: OPENBLAS_NUM_THREADS=1 python3 a6k.py out.json"""
import json, sys, time, statistics
from concurrent.futures import ThreadPoolExecutor
import numpy as np
KVH, G, HD, WIN, W = 5, 4, 128, 1024, 8; SC = np.float32(1 / np.sqrt(HD)); NG = HD // 32; TB_ = 32
CTXS, BS, TKS = (2048, 8192), (1, 8), (1024, 2048)
FMTS = ("a8", "k8v4", "kc4v4", "kc5v4", "kc6v4", "kc5v8", "kc6v8")
WB = {"u32": 16.65, "int8": 66.34}   # MiB, full bitnetd layer weights from 18:55 11/18 run1 (not re-timed)
ROUNDS, REPS = 7, 3
POOL = ThreadPoolExecutor(W)
rng = np.random.default_rng(11200255)
DEC = np.array([[(c & 15) - 8, (c >> 4) - 8] for c in range(256)], np.int8); LUT16 = DEC.view(np.uint16).ravel()
DECU = np.array([[c & 15, c >> 4] for c in range(256)], np.uint8); LUTU16 = DECU.view(np.uint16).ravel()
def q8(x, gs=64):
    g = x.reshape(*x.shape[:-1], HD // gs, gs); m = np.abs(g).max(-1) / 127.0 + 1e-12
    return np.clip(np.rint(g / m[..., None]), -127, 127).astype(np.int8).reshape(x.shape), m.astype(np.float32)
def dq8(c, s): return (c.reshape(*c.shape[:-1], HD // 64, 64).astype(np.float32) * s[..., None]).reshape(c.shape)
def q4(x):
    g = x.reshape(*x.shape[:-1], NG, 32); m = np.abs(g).max(-1) / 7.0 + 1e-12
    c = (np.clip(np.rint(g / m[..., None]), -7, 7).astype(np.int16) + 8).reshape(*x.shape[:-1], HD // 2, 2)
    return (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8), m.astype(np.float32)
def pack4(c):  # (..., n) 0..15 -> (..., n/2)
    n = c.shape[-1]; c = c.astype(np.uint8).reshape(*c.shape[:-1], n // 2, 2); return (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8)
def qcw(x, lim):  # per-channel symmetric over 32-token blocks: x (H,T,HD) -> codes int16 in [-lim,lim], scale (H,T/32,HD)
    H, T, n = x.shape; g = x.reshape(H, T // TB_, TB_, n); m = np.abs(g).max(2) / float(lim) + 1e-12
    return np.clip(np.rint(g / m[:, :, None]), -lim, lim).astype(np.int16).reshape(H, T, n), m.astype(np.float32)
def pack6(c):  # (..., HD) 0..63 -> (..., 3HD/4)
    c = c.astype(np.uint8).reshape(*c.shape[:-1], HD // 4, 4)
    b0 = c[..., 0] | (c[..., 1] << 6); b1 = (c[..., 1] >> 2) | (c[..., 2] << 4); b2 = (c[..., 2] >> 4) | (c[..., 3] << 2)
    return np.stack([b0, b1, b2], -1).astype(np.uint8).reshape(*c.shape[:-2], 3 * HD // 4)
def unpack6(p):  # (T, 3HD/4) -> (T, HD) int8 codes in [-31, 31]
    b = p.reshape(p.shape[0], HD // 4, 3); b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    o = np.empty((p.shape[0], HD // 4, 4), np.uint8)
    np.bitwise_and(b0, 63, out=o[..., 0]); o[..., 1] = (b0 >> 6) | ((b1 & 15) << 2)
    o[..., 2] = (b1 >> 4) | ((b2 & 3) << 4); np.right_shift(b2, 2, out=o[..., 3]); o -= 32
    return o.view(np.int8).reshape(p.shape[0], HD)
def unpack5(lo, hb):  # lo (T, HD/2) nibbles, hb (T, HD/8) bit plane -> (T, HD) int8 in [-15, 15]
    u = np.take(LUTU16, lo).view(np.uint8).reshape(lo.shape[0], HD)
    u |= np.unpackbits(hb, axis=-1, bitorder="little") << 4; u -= 16; return u.view(np.int8)
def kvhist(n):  # same generator as 05:55-18:55 11/19
    ch = np.exp(rng.normal(0, 0.5, (KVH, 1, HD))).astype(np.float32); ch[:, :, rng.choice(HD, 6, replace=False)] *= 6
    k = (rng.standard_normal((KVH, n, HD)).astype(np.float32) + 0.3 * np.sin(np.arange(n) / 300.0)[None, :, None]) * ch
    v = rng.standard_normal((KVH, n, HD)).astype(np.float32) * np.exp(rng.normal(0, 0.3, (KVH, 1, HD))).astype(np.float32)
    return k.astype(np.float32), v.astype(np.float32)
KB = {"kc4": HD // 2, "kc5": HD // 2 + HD // 8, "kc6": 3 * HD // 4}
class KV:
    def __init__(s, k, v):
        s.n = k.shape[1]; s.kf, s.vf = k, v
        (s.k8, s.ks8), (s.v8, s.vs8) = q8(k), q8(v)
        s.v4, s.v4s = q4(dq8(s.v8, s.vs8)); kd = dq8(s.k8, s.ks8)
        c, s.kc4s = qcw(kd, 7); s.kc4 = pack4(c + 8)
        c, s.kc5s = qcw(kd, 15); c = (c + 16).astype(np.uint8); s.kc5lo = pack4(c & 15)
        s.kc5hb = np.packbits((c >> 4).astype(np.uint8), axis=-1, bitorder="little")
        c, s.kc6s = qcw(kd, 31); s.kc6 = pack6(c + 32)
    def bytes(s, fmt):
        n8 = s.n if fmt == "a8" else min(s.n, WIN); n4 = s.n - n8; b8, b4 = HD + HD // 64 * 4, HD // 2 + NG * 4
        if fmt == "a8": return 2 * KVH * s.n * b8
        kold = b8 if fmt.startswith("k8") else KB[fmt[:3]] + HD * 4 // TB_
        vold = b4 if fmt.endswith("v4") else b8
        return KVH * (2 * n8 * b8 + n4 * (kold + vold))
def online(m, l, acc, S, Vfn):
    mn = np.maximum(m, S.max(1, keepdims=True)); al = np.exp(m - mn); S -= mn; np.exp(S, out=S)
    l *= al; l += S.sum(1, keepdims=True); acc *= al; acc += Vfn(S); return mn
def vdq(kv, h, a, b, fmt):
    if fmt.endswith("v8"): return dq8(kv.v8[h, a:b], kv.vs8[h, a:b])
    u = np.take(LUT16, kv.v4[h, a:b]).view(np.int8).reshape(b - a, NG, 32)
    return (u.astype(np.float32) * kv.v4s[h, a:b][:, :, None]).reshape(b - a, HD)
def kdq(kv, h, a, b, fmt):
    T = b - a; a0, b0 = a // TB_, b // TB_; kf = fmt[:3]
    if kf == "k8v": return dq8(kv.k8[h, a:b], kv.ks8[h, a:b])
    if kf == "kc4": u, sc = np.take(LUT16, kv.kc4[h, a:b]).view(np.int8), kv.kc4s
    elif kf == "kc5": u, sc = unpack5(kv.kc5lo[h, a:b], kv.kc5hb[h, a:b]), kv.kc5s
    else: u, sc = unpack6(kv.kc6[h, a:b]), kv.kc6s
    return (u.reshape(T // TB_, TB_, HD).astype(np.float32) * sc[h, a0:b0][:, None]).reshape(T, HD)
def tile_a8(kv, h, a, b, Q, st):
    K = dq8(kv.k8[h, a:b], kv.ks8[h, a:b]); V = dq8(kv.v8[h, a:b], kv.vs8[h, a:b])
    return online(*st, Q @ K.T, lambda P: P @ V)
def tile_old(kv, h, a, b, Q, st, fmt):
    K = kdq(kv, h, a, b, fmt); V = vdq(kv, h, a, b, fmt); return online(*st, Q @ K.T, lambda P: P @ V)
def head(kv, Q, h, fmt, TK):
    st = [np.full((G, 1), -np.inf, np.float32), np.zeros((G, 1), np.float32), np.zeros((G, HD), np.float32)]
    wb = 0 if fmt == "a8" else max(0, kv.n - WIN)
    for a in range(0, wb, TK): st[0] = tile_old(kv, h, a, min(a + TK, wb), Q, st, fmt)
    for a in range(wb, kv.n, TK if fmt == "a8" else WIN):
        st[0] = tile_a8(kv, h, a, min(a + (TK if fmt == "a8" else WIN), kv.n), Q, st)
    return st[2] / st[1]
def attn(kvs, q, fmt, TK):
    futs = [POOL.submit(head, kvs[s], q[s, h], h, fmt, TK) for s in range(len(kvs)) for h in range(KVH)]
    return np.stack([f.result() for f in futs])
def ref(kv, Q, h):
    S = Q @ kv.kf[h].T; S -= S.max(1, keepdims=True); P = np.exp(S); P /= P.sum(1, keepdims=True); return P @ kv.vf[h]
def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
def main():
    out = {"rows": [], "compare": {}, "unpack": {}}
    print(f"{'cfg':28s} {'attn ms med (min-max)':>24s} {'KV MiB':>8s} {'min cos':>10s} {'K rel err':>10s}")
    for n in CTXS:
        kvs8 = [KV(*kvhist(n)) for _ in range(max(BS))]
        q8all = (rng.standard_normal((max(BS), KVH, G, HD)).astype(np.float32) * SC)
        # old-token K reconstruction error (rel Frobenius vs f32 K), per K format
        wb = n - WIN
        for kf in ("k8v", "kc4", "kc5", "kc6"):
            e = [np.linalg.norm(kdq(kv, h, 0, wb, kf) - kv.kf[h, :wb]) / np.linalg.norm(kv.kf[h, :wb]) for kv in kvs8 for h in range(KVH)]
            out["compare"][f"Kerr_{kf}_ctx{n}"] = float(np.median(e)); print(f"  old-K rel err {kf} ctx{n}: median {np.median(e):.5f} max {max(e):.5f}")
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
                    print(f"{key:28s} {r['med']:8.2f} ({r['lo']:.2f}-{r['hi']:.2f})      {r['mib']:8.2f} {c:10.7f}")
            best = {f: min((res[(f, t)] for t in TKS), key=lambda r: r["med"]) for f in FMTS}
            for f in ("kc5v4", "kc6v4", "kc4v4", "kc5v8", "kc6v8"):
                for refn in ("a8", "k8v4"):
                    k = f"{f}_vs_{refn}_ctx{n}_B{B}"; r = best[refn]["med"] / best[f]["med"]
                    out["compare"][k] = dict(speed=r, bytes=best[refn]["mib"] / best[f]["mib"], dcos=best[f]["cos"] - best[refn]["cos"])
                    print(f"  {k:26s} speed {r:5.2f}x  KV bytes {best[refn]['mib'] / best[f]['mib']:5.2f}x  dcos {best[f]['cos'] - best[refn]['cos']:+.5f}")
            if B == 8:
                for wf in ("u32", "int8"):
                    for f in FMTS: out["compare"][f"total_{wf}_{f}_ctx{n}_B8"] = WB[wf] + best[f]["mib"]
                t = lambda wf, f: out["compare"][f"total_{wf}_{f}_ctx{n}_B8"]
                for f in FMTS:
                    print(f"  total ctx{n} B8 u32+{f}: {t('u32', f):7.2f} MiB | vs int8+a8 {t('int8','a8')/t('u32', f):5.2f}x | vs int8+same {t('int8', f)/t('u32', f):5.2f}x | vs u32+k8v4 {t('u32','k8v4')/t('u32', f):5.2f}x")
    # isolated K unpack+dequant cost for all old tokens of one decode step at B=8 / ctx 8192 (single thread, 40 heads)
    kvs = kvs8; wb = 8192 - WIN
    for kf in ("k8v", "kc4", "kc5", "kc6"):
        ts = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            for kv in kvs:
                for h in range(KVH):
                    for a in range(0, wb, 2048): kdq(kv, h, a, min(a + 2048, wb), kf)
            ts.append((time.perf_counter() - t0) * 1e3)
        out["unpack"][kf] = dict(med=statistics.median(ts), lo=min(ts), hi=max(ts))
        print(f"  K unpack+dequant {kf} B8 ctx8192 (1 thread, 7168 old tok x 40 heads): {statistics.median(ts):.2f} ms ({min(ts):.2f}-{max(ts):.2f})")
    json.dump(out, open(sys.argv[1], "w"), indent=1)
main()
```

## Next question

V at A4 is now the limit: 0.0058 of min cos at B = 8 / ctx 8192, against 0.0007 for A6 K. V's error can be rotated away almost for free, because P·(V H) = (P·V) H. So a per-head Hadamard rotation of V before A4 g32 costs one G × HD × HD matmul per head per step to undo, and nothing per token.

Does Hadamard-rotated A4 g32 V (with K8, kc6 or kc5 K) lift K8V4's min cos toward all-A8 (≥ 0.995)? Does it let kc5 K + rotated A4 V reach ≥ K8V4 − 0.002 at the 1.15x total-byte saving vs K8V4? Also measure the write-side rotation cost per new token and the per-step un-rotate cost at B = 8.

close: sha256(prev || this)[:16] = 02eff14c852cb14b
