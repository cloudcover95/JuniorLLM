---
local: true
date: 2026-11-19
wave: "2026-11-19 14:55 (ran ~14:40–14:50 MT; Chief last_run 2026-11-19T21:05:00Z = 14:05 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-stream, kv-cache, kivi, per-channel, k4v4, k8v4, a4, a8-g64, recent-window, gqa, numpy, not-viable]
---

# KIVI-style per-channel A4 K (scale per channel over 32-token blocks) lifts K4V4 min attention-output cos at B = 8 / ctx 8192 from **0.735 to 0.938** (symmetric) / **0.957** (asymmetric, scale + zero), but neither reaches the 0.99 bar (K8V4 0.989, all-A8 0.995 on this seed). Bytes: symmetric = K4V4 (71.03 MiB total with 2-bit weights, 2.13x below int8 + A8), asymmetric 75.40 MiB (2.01x). Attention speed is on par with K4V4/K8V4 (0.90–1.16x of K8V4 at B = 8 / ctx 8192 across the two runs). Not viable: same verdict as the 10:55 Hadamard rotation, accuracy short of K8V4. Keep K8V4.

## Question

[[2026-11-19_1055_hadamard-rotated-k-before-a4-g32-k4v4-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-and-all-a8-u32-lut-total-bytes]] found that rotating K stops at min cos 0.906. Its next question: does KIVI-style per-channel K quantization at A4 (one scale per channel over a 32-token block; V stays per-token A4 g32) reach min cos ≥ 0.99 at ≤ K8V4 bytes, and what does the per-channel scale layout cost in numpy decode attention at B = 8 / ctx 8192?

## Method

- Same attention-only microbench and synthetic K/V generator as 05:55/10:55 (`hadk4.py` → `kivi4.py`): Gaussian K/V with log-normal per-channel spread, 6 hot K channels × 6, slow positional drift (0.3 · sin(t/300) added before the channel scale); new seed 11191455. B ∈ {1, 8} streams × 5 KV heads × 4 q heads (GQA 20/5), HD 128, ctx ∈ {2048, 8192}, last 1024 tokens at A8 g64 (they double as the staging buffer that fills a 32-token block before it ages out). Online softmax per (stream, KV head) on a W = 8 thread pool, `OPENBLAS_NUM_THREADS=1`, TK ∈ {1024, 2048}, best TK per variant used.
- New variants for the old (aged-out) K region:
  - `kc4v4`: symmetric per-channel A4, codes [-7, 7], f32 scale per (32-token block, channel); 64 B codes + 16 B scales per token (same as K4V4 g32).
  - `kc4av4`: asymmetric per-channel A4 (KIVI), codes 0..15, f32 scale + f32 zero per (block, channel); 96 B per token.
  - Decode path: u16 LUT → codes → f32 × block scale (+ zero) broadcast over the 32 rows → Q Kᵀ, P V (V dequant as K4V4).
- Controls: `a8` (all A8 g64), `a4deq` (per-token A4 g32 K4V4), `k8v4` (05:55 recommendation).
- Write side: per-token g32 quantizes one aged-out K row per stream per KV head every step; per-channel quantizes a 32-row block every 32 steps (timed per block, divided by 32). 200 iterations × 7 rounds.
- Total bytes = full bitnetd layer weights from 18:55 11/18 run1 (u32 LUT 16.65 MiB, int8 66.34 MiB; not re-timed) + measured KV at B = 8.
- Error: min over (stream, head) of cos(attention output, exact f32-KV softmax). 7 rounds × 3 reps per config, median (min–max), two independent runs (same data and error, timing differs).

## Command

```
cd /workspace/bench1455_1119
OPENBLAS_NUM_THREADS=1 python3 kivi4.py results_run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 python3 kivi4.py results_run2.json > run2.txt
```

## Results: attention ms per decode step, all streams (run1 / run2, best TK, median)

| ctx / B | a8 | a4deq (K4V4) | k8v4 | kc4v4 | kc4av4 | KV MiB a8 / K4V4 = kc4v4 / kc4av4 / K8V4 | kc4v4 vs k8v4 | kc4av4 vs k8v4 |
|---|---|---|---|---|---|---|---|---|
| 2048 / 1 | 1.37 / 1.05 | 1.79 / 1.50 | 1.39 / 1.41 | 1.36 / 1.44 | 1.49 / 1.49 | 2.66 / 2.11 / 2.19 / 2.38 | 1.02x / 0.98x | 0.93x / 0.95x |
| 2048 / 8 | 6.26 / 7.52 | 10.03 / 12.89 | 10.24 / 12.03 | 11.49 / 13.15 | 13.14 / 13.50 | 21.25 / 16.88 / 17.50 / 19.06 | 0.89x / 0.91x | 0.78x / 0.89x |
| 8192 / 1 | 4.17 / 3.41 | 4.95 / 4.40 | 4.31 / 3.74 | 5.10 / 4.47 | 5.88 / 4.86 | 10.62 / 6.80 / 7.34 / 8.71 | 0.85x / 0.84x | 0.73x / 0.77x |
| 8192 / 8 | 25.42 / 25.15 | 35.75 / 31.03 | 36.98 / 34.63 | 38.31 / 29.84 | 39.95 / 29.66 | 85.00 / 54.38 / 58.75 / 69.69 | 0.97x / 1.16x | 0.93x / 1.17x |

Spread run1 at B = 8 / ctx 8192 (best TK): a8 24.57–26.44, a4deq 35.33–37.35, k8v4 34.26–38.98, kc4v4 35.32–41.03, kc4av4 37.76–47.48 ms. Run2: a8 24.30–27.70, a4deq 30.25–32.60, k8v4 29.72–35.00, kc4v4 29.27–31.31, kc4av4 29.15–31.74 ms.

Error (min attention-output cos; identical in both runs):

| ctx / B | a8 | a4deq (K4V4) | k8v4 | kc4v4 | kc4av4 |
|---|---|---|---|---|---|
| 2048 / 1 | 0.99805 | 0.89971 | 0.99405 | 0.97401 | 0.98413 |
| 2048 / 8 | 0.99706 | 0.66321 | 0.99207 | 0.97401 | 0.97839 |
| 8192 / 1 | 0.99892 | 0.73492 | 0.99202 | 0.97783 | 0.98250 |
| 8192 / 8 | 0.99462 | 0.73492 | 0.98913 | 0.93763 | 0.95679 |

Write-side age-out cost per decode step (run1 / run2 median): B = 1 g32 row 0.0187 / 0.0169 ms, per-channel sym block/32 0.0019 / 0.0016 ms, asym 0.0022 / 0.0019 ms; B = 8 g32 row 0.0382 / 0.0348 ms, sym 0.0138 / 0.0118 ms, asym 0.0153 / 0.0136 ms. Batching 32 rows makes the write side ~2.5–10x cheaper, but it is < 0.2% of a B = 8 / ctx 8192 step either way.

Total bytes per layer at B = 8 (weights + KV, MiB):

| ctx | u32 + a8 | u32 + K4V4 = u32 + kc4v4 | u32 + kc4av4 | u32 + K8V4 | int8 + a8 | kc4v4 vs int8+a8 | kc4av4 vs int8+a8 | K8V4 vs int8+a8 |
|---|---|---|---|---|---|---|---|---|
| 2048 | 37.90 | 33.52 | 34.15 | 35.71 | 87.59 | 2.61x | 2.56x | 2.45x |
| 8192 | 101.65 | 71.03 | 75.40 | 86.34 | 151.34 | 2.13x | 2.01x | 1.75x |

## Verdict

**not-viable.** Per-channel K scaling is the better A4 K layout here (min cos 0.938–0.957 vs 0.735 per-token g32 at B = 8 / ctx 8192, beating the 10:55 Hadamard result of 0.906 on its seed), costs nothing on the read side in numpy (within ±17% of K8V4, faster than K8V4 in run2) and is cheaper on the write side. But the accuracy still falls 0.03–0.05 short of K8V4 and degrades with context and stream count, so the extra total-bytes edge (2.01–2.13x vs 1.75x below int8 + A8, i.e. 1.15–1.22x fewer bytes than K8V4) does not come at acceptable error. K8V4 stays the recommended old-token cache; the 2-bit ternary weights still carry the memory win either way. Caveat: synthetic K with fixed hot channels; on this seed K8V4 itself sits at 0.989.

## Links

- previous: [[2026-11-19_1055_hadamard-rotated-k-before-a4-g32-k4v4-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-and-all-a8-u32-lut-total-bytes]]
- related: [[2026-11-19_0555_k8v4-a8-g64-keys-all-tokens-a4-g32-old-values-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k4v4-and-all-a8-u32-lut-total-bytes]], [[2026-11-18_1855_a4-g32-nibble-old-kv-plus-a8-g64-recent-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-u32-lut-vs-int8-weights-total-bytes-full-bitnetd-layer]]
- this wave's pair files: none. Quant Chief 14:05 assignment was pair_A no gap and pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-19T21:05:00Z). Roadmap blobs: `JuniorHome docs/ROADMAP.md` 4efb666, `JuniorLLM docs/BETA_TO_OS.md` bac6923.
- bench dir: `/workspace/bench1455_1119` (kivi4.py, run1.txt, run2.txt, results_run1.json, results_run2.json)

## Script (`kivi4.py`)

```python
#!/usr/bin/env python3
"""KIVI-style per-channel A4 K for old tokens (14:55 11/19). Derived from hadk4.py (10:55 11/19).
Aged-out K rows are quantized per CHANNEL over 32-token blocks (one scale per (block, channel)), instead of per token
over 32-dim groups. V stays per-token A4 g32. The last WIN=1024 tokens stay A8 g64 (they also serve as the staging
buffer that fills a 32-token block before it is quantized).
New variants:
  kc4v4  : symmetric per-channel A4 K ([-7,7], f32 scale per (32-tok block, channel)) + A4 g32 V  (same bytes as K4V4)
  kc4av4 : asymmetric per-channel A4 K (codes 0..15, f32 scale + f32 zero per (block, channel), KIVI) + A4 g32 V
Controls: a8 (all A8 g64), a4deq (per-token A4 g32 K4V4), k8v4 (05:55 recommendation: A8 K all tokens + A4 g32 old V).
Attention-only microbench: B streams x 5 KV heads x 4 q heads (GQA 20/5), HD 128, online softmax per (stream, KV head)
over TK-key tiles on a W-thread pool. Error: min cos of attention output vs exact f32-KV softmax.
Usage: OPENBLAS_NUM_THREADS=1 python3 kivi4.py out.json"""
import json, sys, time, statistics
from concurrent.futures import ThreadPoolExecutor
import numpy as np
KVH, G, HD, WIN, W = 5, 4, 128, 1024, 8; SC = np.float32(1 / np.sqrt(HD)); NG = HD // 32
CTXS, BS, TKS = (2048, 8192), (1, 8), (1024, 2048)
FMTS = ("a8", "a4deq", "k8v4", "kc4v4", "kc4av4")
WB = {"u32": 16.65, "int8": 66.34}   # MiB, full bitnetd layer weights from 18:55 11/18 run1
ROUNDS, REPS = 7, 3
POOL = ThreadPoolExecutor(W)
rng = np.random.default_rng(11191455); TB_ = 32
DEC = np.array([[(c & 15) - 8, (c >> 4) - 8] for c in range(256)], np.int8)
LUT16 = DEC.view(np.uint16).ravel(); LUTF = DEC.astype(np.float32)          # (256, 2) f32 pair table
def q8(x, gs=64):
    g = x.reshape(*x.shape[:-1], HD // gs, gs); m = np.abs(g).max(-1) / 127.0 + 1e-12
    return np.clip(np.rint(g / m[..., None]), -127, 127).astype(np.int8).reshape(x.shape), m.astype(np.float32)
def dq8(c, s): return (c.reshape(*c.shape[:-1], HD // 64, 64).astype(np.float32) * s[..., None]).reshape(c.shape)
def q4(x):
    g = x.reshape(*x.shape[:-1], NG, 32); m = np.abs(g).max(-1) / 7.0 + 1e-12
    c = (np.clip(np.rint(g / m[..., None]), -7, 7).astype(np.int16) + 8).reshape(*x.shape[:-1], HD // 2, 2)
    return (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8), m.astype(np.float32)
DECU = np.array([[c & 15, c >> 4] for c in range(256)], np.uint8); LUTU16 = DECU.view(np.uint16).ravel()
def pack(c):  # (..., HD) small ints in 0..15 -> (..., HD/2) bytes
    c = c.astype(np.uint8).reshape(*c.shape[:-1], HD // 2, 2); return (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8)
def qc4(x):   # per-channel symmetric: x (H, T, HD), T % 32 == 0 -> codes (H,T,HD/2), scale (H,T/32,HD)
    H, T, _ = x.shape; g = x.reshape(H, T // TB_, TB_, HD); m = np.abs(g).max(2) / 7.0 + 1e-12
    c = np.clip(np.rint(g / m[:, :, None]), -7, 7).astype(np.int16) + 8
    return pack(c.reshape(H, T, HD)), m.astype(np.float32)
def qc4a(x):  # per-channel asymmetric (KIVI): codes 0..15, scale + zero per (block, channel)
    H, T, _ = x.shape; g = x.reshape(H, T // TB_, TB_, HD); lo = g.min(2); hi = g.max(2); m = (hi - lo) / 15.0 + 1e-12
    c = np.clip(np.rint((g - lo[:, :, None]) / m[:, :, None]), 0, 15)
    return pack(c.reshape(H, T, HD)), m.astype(np.float32), lo.astype(np.float32)
def kvhist(n):  # realistic-ish K/V: per-channel scale spread + a few hot K channels, slow drift over position
    ch = np.exp(rng.normal(0, 0.5, (KVH, 1, HD))).astype(np.float32); ch[:, :, rng.choice(HD, 6, replace=False)] *= 6
    k = (rng.standard_normal((KVH, n, HD)).astype(np.float32) + 0.3 * np.sin(np.arange(n) / 300.0)[None, :, None]) * ch
    v = rng.standard_normal((KVH, n, HD)).astype(np.float32) * np.exp(rng.normal(0, 0.3, (KVH, 1, HD))).astype(np.float32)
    return k.astype(np.float32), v.astype(np.float32)
class KV:
    def __init__(s, k, v):
        s.n = k.shape[1]; s.kf, s.vf = k, v
        (s.k8, s.ks8), (s.v8, s.vs8) = q8(k), q8(v)
        (s.k4, s.k4s), (s.v4, s.v4s) = q4(dq8(s.k8, s.ks8)), q4(dq8(s.v8, s.vs8))   # aged out of the A8 window
        kd = dq8(s.k8, s.ks8)
        s.kc4, s.kc4s = qc4(kd); s.kca, s.kcas, s.kcaz = qc4a(kd)                      # per-channel, 32-token blocks
    def bytes(s, fmt):
        n8 = s.n if fmt == "a8" else min(s.n, WIN); n4 = s.n - n8; b8, b4 = HD + HD // 64 * 4, HD // 2 + NG * 4
        if fmt == "k8v4": return KVH * (s.n * b8 + n8 * b8 + n4 * b4)
        if fmt == "kc4av4": return KVH * (2 * n8 * b8 + n4 * b4 + n4 * (HD // 2 + HD * 4 // TB_ * 2))
        return 2 * KVH * (n8 * b8 + n4 * b4)
def online(m, l, acc, S, Vfn):
    mn = np.maximum(m, S.max(1, keepdims=True)); al = np.exp(m - mn); S -= mn; np.exp(S, out=S)
    l *= al; l += S.sum(1, keepdims=True); acc *= al; acc += Vfn(S); return mn
def tile_a8(kv, h, a, b, Q, st):
    K = dq8(kv.k8[h, a:b], kv.ks8[h, a:b]); V = dq8(kv.v8[h, a:b], kv.vs8[h, a:b])
    return online(*st, Q @ K.T, lambda P: P @ V)
def tile_a4deq(kv, h, a, b, Q, st):
    def dq4(c, sc):
        u = np.take(LUT16, c).view(np.int8).reshape(b - a, NG, 32)
        return (u.astype(np.float32) * sc[:, :, None]).reshape(b - a, HD)
    K = dq4(kv.k4[h, a:b], kv.k4s[h, a:b]); V = dq4(kv.v4[h, a:b], kv.v4s[h, a:b])
    return online(*st, Q @ K.T, lambda P: P @ V)
def tile_kc4(kv, h, a, b, Q, st, asym):
    T = b - a; a0, b0 = a // TB_, b // TB_
    if asym:
        u = np.take(LUTU16, kv.kca[h, a:b]).view(np.uint8).reshape(T // TB_, TB_, HD)
        K = (u.astype(np.float32) * kv.kcas[h, a0:b0][:, None] + kv.kcaz[h, a0:b0][:, None]).reshape(T, HD)
    else:
        u = np.take(LUT16, kv.kc4[h, a:b]).view(np.int8).reshape(T // TB_, TB_, HD)
        K = (u.astype(np.float32) * kv.kc4s[h, a0:b0][:, None]).reshape(T, HD)
    w = np.take(LUT16, kv.v4[h, a:b]).view(np.int8).reshape(T, NG, 32)
    V = (w.astype(np.float32) * kv.v4s[h, a:b][:, :, None]).reshape(T, HD)
    return online(*st, Q @ K.T, lambda P: P @ V)
def tile_k8v4(kv, h, a, b, Q, st):
    K = dq8(kv.k8[h, a:b], kv.ks8[h, a:b])
    u = np.take(LUT16, kv.v4[h, a:b]).view(np.int8).reshape(b - a, NG, 32)
    V = (u.astype(np.float32) * kv.v4s[h, a:b][:, :, None]).reshape(b - a, HD)
    return online(*st, Q @ K.T, lambda P: P @ V)
def vfold(kv, h, a, b):
    Vc = np.take(LUTF, kv.v4[h, a:b], axis=0).reshape(b - a, NG, 32); vs = kv.v4s[h, a:b]   # (T,4,32), (T,4)
    def f(P):   # sum_t P[q,t] vs[t,g] Vc[t,g,:]
        out = np.empty((P.shape[0], NG, 32), np.float32)
        for g in range(NG): out[:, g] = (P * vs[:, g]) @ Vc[:, g]
        return out.reshape(P.shape[0], HD)
    return f
def tile_a4fold(kv, h, a, b, Q, st, Qbd):
    Kc = np.take(LUTF, kv.k4[h, a:b], axis=0).reshape(b - a, HD)          # raw codes as f32, no scale multiply
    Sg = (Qbd @ Kc.T).reshape(NG, G, b - a)                               # per-group partial scores
    S = np.einsum("gqt,tg->qt", Sg, kv.k4s[h, a:b], optimize=False)
    return online(*st, np.ascontiguousarray(S), vfold(kv, h, a, b))
def tile_a4byte(kv, h, a, b, Q, st, TB):
    idx = kv.k4[h, a:b].astype(np.int32) + OFF                            # (T, 64) flat index into (64*256)
    Sp = np.take(TB, idx, axis=0)                                         # (T, 64, G)
    Sg = Sp.reshape(b - a, NG, 16, G).sum(2)                              # (T, 4, G)
    S = np.einsum("tgq,tg->qt", Sg, kv.k4s[h, a:b], optimize=False)
    return online(*st, np.ascontiguousarray(S), vfold(kv, h, a, b))
OFF = (np.arange(HD // 2, dtype=np.int32) * 256)[None, :]
def head(kv, Q, h, fmt, TK):
    st = [np.full((G, 1), -np.inf, np.float32), np.zeros((G, 1), np.float32), np.zeros((G, HD), np.float32)]
    wb = 0 if fmt == "a8" else max(0, kv.n - WIN)
    if fmt == "a4fold":
        Qbd = np.zeros((NG, G, HD), np.float32)
        for g in range(NG): Qbd[g, :, g * 32:(g + 1) * 32] = Q[:, g * 32:(g + 1) * 32]
        Qbd = Qbd.reshape(NG * G, HD)
    if fmt == "a4byte":   # table[pos, byte, q] = Q[q, 2pos]*dec0 + Q[q, 2pos+1]*dec1
        Qp = Q.reshape(G, HD // 2, 2); TB = np.einsum("qpc,bc->pbq", Qp, LUTF).reshape(HD // 2 * 256, G).copy()
    for a in range(0, wb, TK):
        b = min(a + TK, wb)
        if fmt == "a4deq": st[0] = tile_a4deq(kv, h, a, b, Q, st)
        elif fmt == "k8v4": st[0] = tile_k8v4(kv, h, a, b, Q, st)
        elif fmt == "kc4v4": st[0] = tile_kc4(kv, h, a, b, Q, st, False)
        elif fmt == "kc4av4": st[0] = tile_kc4(kv, h, a, b, Q, st, True)
        elif fmt == "a4fold": st[0] = tile_a4fold(kv, h, a, b, Q, st, Qbd)
        elif fmt == "a4byte": st[0] = tile_a4byte(kv, h, a, b, Q, st, TB)
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
    out = {"rows": [], "compare": {}}
    print(f"{'cfg':28s} {'attn ms med (min-max)':>24s} {'KV MiB':>8s} {'min cos':>10s} {'max|d| vs a4deq':>15s}")
    for n in CTXS:
        kvs8 = [KV(*kvhist(n)) for _ in range(max(BS))]
        q8all = (rng.standard_normal((max(BS), KVH, G, HD)).astype(np.float32) * SC)
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
                    res[(fmt, TK)] = dict(key=key, med=statistics.median(ts), lo=min(ts), hi=max(ts), ts=ts,
                                          mib=B * kvs[0].bytes(fmt) / 2**20, cos=c, o=o)
            for (fmt, TK), r in res.items(): r["maxdiff_vs_a4deq"] = float(np.abs(r["o"] - res[("a4deq", TK)]["o"]).max())
            for (fmt, TK), r in res.items():
                d = r["maxdiff_vs_a4deq"]; r.pop("o"); out["rows"].append(r)
                print(f"{r['key']:28s} {r['med']:8.2f} ({r['lo']:.2f}-{r['hi']:.2f})      {r['mib']:8.2f} {r['cos']:10.7f} {d:15.2e}")
            best = {f: min((res[(f, t)] for t in TKS), key=lambda r: r["med"]) for f in FMTS}
            for f in ("kc4v4", "kc4av4", "k8v4", "a4deq"):
                for refn in ("a4deq", "a8", "k8v4"):
                    if f == refn: continue
                    k = f"{f}_vs_{refn}_ctx{n}_B{B}"; r = best[refn]["med"] / best[f]["med"]
                    out["compare"][k] = dict(speed=r, bytes=best[refn]["mib"] / best[f]["mib"], new=best[f]["key"], ref=best[refn]["key"])
                    print(f"  {k:30s} speed {r:5.2f}x  bytes {best[refn]['mib'] / best[f]['mib']:5.2f}x  ({best[f]['key']} vs {best[refn]['key']})")
            if B == 8:   # total-bytes (weights + KV) per layer, MiB
                for wf in ("u32", "int8"):
                    for f in FMTS: out["compare"][f"total_{wf}_{f}_ctx{n}_B8"] = WB[wf] + best[f]["mib"]
                t = lambda wf, f: out["compare"][f"total_{wf}_{f}_ctx{n}_B8"]
                for f in FMTS:
                    print(f"  total ctx{n} B8 u32+{f}: {t('u32', f):7.2f} MiB | vs int8+a8 {t('int8','a8')/t('u32', f):5.2f}x | vs int8+same {t('int8', f)/t('u32', f):5.2f}x | vs u32+a8 {t('u32','a8')/t('u32', f):5.2f}x")
    # write-side: per step each stream ages one K row per KV head out of the window. per-token g32: quantize that row now;
    # per-channel: every 32 steps quantize a 32-row block (cost amortized /32). Times per step (ms).
    for B in BS:
        kr = rng.standard_normal((B, KVH, TB_, HD)).astype(np.float32); c8, s8 = q8(kr)
        jobs = (("g32_row", lambda: q4(dq8(c8[:, :, :1], s8[:, :, :1])), 1),
                ("pc_sym_block", lambda: qc4(dq8(c8, s8).reshape(B * KVH, TB_, HD)), TB_),
                ("pc_asym_block", lambda: qc4a(dq8(c8, s8).reshape(B * KVH, TB_, HD)), TB_))
        for name, fn, div in jobs:
            ts = []
            for _ in range(ROUNDS):
                t0 = time.perf_counter()
                for _ in range(200): fn()
                ts.append((time.perf_counter() - t0) / 200 * 1e3 / div)
            out["compare"][f"write_{name}_B{B}"] = dict(med=statistics.median(ts), lo=min(ts), hi=max(ts))
            print(f"  write-side per step B{B} {name}: {statistics.median(ts):.4f} ms ({min(ts):.4f}-{max(ts):.4f})")
    json.dump(out, open(sys.argv[1], "w"), indent=1)
main()
```

## Next question

Both A4 K fixes (rotation, per-channel) stall at 0.91–0.96 while K8V4 holds ~0.99. Is the K-side error dominated by the few hot channels? Test a mixed K layout for old tokens: keep the top-k highest-range K channels per head (k ∈ {4, 8, 16}, picked from the A8 window) at A8 and the rest at per-channel A4, with V at A4 g32. Does it reach min cos ≥ 0.99 at fewer bytes than K8V4 (≥ 1.15x KV saving), and what does the channel gather cost per decode step at B = 8 / ctx 8192?

close: sha256(prev || this)[:16] = 0060a9c3396398a6
