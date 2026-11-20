---
local: true
date: 2026-11-19
wave: "2026-11-19 18:55 (ran ~18:45–18:55 MT; Chief last_run 2026-11-20T01:09:00Z = 18:09 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-stream, kv-cache, mixed-precision, hot-channels, kivi, per-channel, k8v4, a4, a8-g64, recent-window, gqa, numpy, not-viable]
---

# Keeping the top-k highest-range K channels per head at A8 and the rest at per-channel A4 lifts old-token K accuracy at B = 8 / ctx 8192 from min cos **0.959** (KIVI sym, k = 0) to **0.976 / 0.976 / 0.982** (k = 4 / 8 / 16), but never reaches K8V4 (**0.988**) or the 0.99 bar. Bytes stay close to K4V4 (71.3–73.2 MiB total with 2-bit weights, 2.07–2.12x below int8 + A8; 1.18–1.21x below K8V4), and attention runs at 0.73–0.83x of K8V4. Not viable: hot channels are not the whole K error. Keep K8V4.

## Question

[[2026-11-19_1455_kivi-per-channel-a4-k-sym-and-asym-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k4v4-k8v4-and-all-a8-u32-lut-total-bytes]] found per-channel A4 K stalls at 0.94–0.96. Its next question: is the K-side error dominated by the few hot channels? Keep the top-k highest-range K channels per head (k ∈ {4, 8, 16}, picked from the A8 window) at A8 and the rest at per-channel A4, V at A4 g32. Does it reach min cos ≥ 0.99 at fewer bytes than K8V4 (≥ 1.15x KV saving), and what does the channel gather cost per decode step at B = 8 / ctx 8192?

## Method

- Same attention-only microbench and synthetic K/V generator as 05:55/10:55/14:55 (`kivi4.py` → `mixk.py`): Gaussian K/V, log-normal per-channel spread, 6 planted hot K channels × 6 per head, slow positional drift; new seed 11191855. B ∈ {1, 8} streams × 5 KV heads × 4 q heads (GQA 20/5), HD 128, ctx ∈ {2048, 8192}, last 1024 tokens at A8 g64. Online softmax per (stream, KV head) on a W = 8 thread pool, `OPENBLAS_NUM_THREADS=1`, TK ∈ {1024, 2048}, best TK per variant.
- New variants `mk4` / `mk8` / `mk16` for aged-out K: per KV head rank channels by range (max − min) over the A8 window, permute hot-first once. Hot k channels: symmetric per-channel int8, f32 scale per (32-token block, channel). Cold HD − k channels: symmetric per-channel A4 (codes [−7, 7], nibble-packed), f32 scale per (block, channel). Per old token: k + k/8 + (HD − k)/2 + (HD − k)/8 bytes, plus a 256 B int16 permutation per head.
- Decode: permute Q once per head per step (`Q[:, perm]`, split hot/cold), then S = Q_c K_cᵀ + Q_h K_hᵀ per tile; V dequant as K4V4.
- Controls: `a8` (all A8 g64), `k8v4` (05:55 recommendation), `kc4v4` (14:55 KIVI sym, = k 0).
- Side costs: hot-channel selection (range + argsort over a 1024-token window for all B × 5 heads) and per-step Q permute + split; 7 rounds × 20 / 200 iterations.
- Total bytes = full bitnetd layer weights from 18:55 11/18 run1 (u32 LUT 16.65 MiB, int8 66.34 MiB; not re-timed) + measured KV at B = 8.
- Error: min over (stream, head) of cos(attention output, exact f32-KV softmax). 7 rounds × 3 reps per config, median (min–max), two independent runs (identical error, timing differs).

## Command

```
cd /workspace/bench1855_1119
OPENBLAS_NUM_THREADS=1 python3 mixk.py results_run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 python3 mixk.py results_run2.json > run2.txt
```

## Results: attention ms per decode step, all streams (run1 / run2, best TK, median)

| ctx / B | a8 | k8v4 | kc4v4 (k 0) | mk4 | mk8 | mk16 | KV MiB a8 / k8v4 / kc4v4 / mk4 / mk8 / mk16 |
|---|---|---|---|---|---|---|---|
| 2048 / 1 | 0.98 / 1.02 | 1.21 / 1.39 | 1.30 / 1.35 | 1.48 / 1.49 | 1.46 / 1.68 | 1.86 / 1.41 | 2.66 / 2.38 / 2.11 / 2.12 / 2.13 / 2.15 |
| 2048 / 8 | 5.92 / 5.89 | 9.04 / 9.25 | 9.95 / 9.80 | 11.77 / 11.26 | 11.06 / 10.90 | 11.82 / 10.85 | 21.25 / 19.06 / 16.88 / 16.92 / 17.04 / 17.20 |
| 8192 / 1 | 3.11 / 2.99 | 3.85 / 4.34 | 4.12 / 4.61 | 4.87 / 5.45 | 4.88 / 5.46 | 4.72 / 4.64 | 10.62 / 8.71 / 6.80 / 6.83 / 6.93 / 7.07 |
| 8192 / 8 | 19.71 / 19.69 | 25.58 / 26.33 | 27.30 / 30.86 | 32.64 / 35.11 | 31.57 / 31.65 | 32.14 / 36.16 | 85.00 / 69.69 / 54.38 / 54.66 / 55.48 / 56.57 |

Spread at B = 8 / ctx 8192 (best TK, run1 | run2): a8 18.81–20.34 | 18.68–20.75, k8v4 24.64–26.51 | 25.77–29.52, kc4v4 26.90–28.80 | 29.04–32.11, mk4 31.27–34.22 | 34.60–36.85, mk8 30.72–33.15 | 31.08–34.03, mk16 30.96–36.29 | 35.67–36.65 ms. Speed vs k8v4 at B = 8 / ctx 8192: mk4 0.78x / 0.75x, mk8 0.81x / 0.83x, mk16 0.80x / 0.73x.

Error (min attention-output cos; identical in both runs):

| ctx / B | a8 | k8v4 | kc4v4 (k 0) | mk4 | mk8 | mk16 |
|---|---|---|---|---|---|---|
| 2048 / 1 | 0.99759 | 0.99462 | 0.96919 | 0.97767 | 0.97419 | 0.98170 |
| 2048 / 8 | 0.99653 | 0.99329 | 0.96886 | 0.97767 | 0.97419 | 0.98170 |
| 8192 / 1 | 0.99553 | 0.99150 | 0.97768 | 0.98278 | 0.98182 | 0.98491 |
| 8192 / 8 | 0.99423 | 0.98788 | 0.95916 | 0.97606 | 0.97581 | 0.98169 |

Side costs (run1 / run2 median): Q permute + split per decode step B = 1 0.0148 / 0.0143 ms, B = 8 0.118 / 0.119 ms (≈ 0.4% of a B = 8 / ctx 8192 step). Hot-channel selection over the 1024-token window B = 1 0.34 / 0.32 ms, B = 8 2.32 / 2.87 ms; done once here (static channel set), but refreshing it every 32-token block would add ~0.07–0.09 ms per step at B = 8. The measured attention slowdown vs kc4v4 (0.84–0.98x) comes from the two-GEMM split and the extra int8 block dequant, not the Q gather.

Total bytes per layer at B = 8 (weights + KV, MiB):

| ctx | u32 + a8 | u32 + K8V4 | u32 + kc4v4 | u32 + mk4 | u32 + mk8 | u32 + mk16 | int8 + a8 | mk16 vs int8+a8 | mk16 vs u32+K8V4 |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 37.90 | 35.71 | 33.52 | 33.57 | 33.69 | 33.85 | 87.59 | 2.59x | 1.06x |
| 8192 | 101.65 | 86.34 | 71.03 | 71.31 | 72.13 | 73.22 | 151.34 | 2.07x | 1.18x |

## Verdict

**not-viable.** Promoting hot channels to A8 recovers part of the per-channel A4 K gap (min cos 0.959 → 0.982 at B = 8 / ctx 8192 with k = 16) at ~1–4% more KV bytes, but even k = 16 stays 0.006 below K8V4 and 0.008 below the 0.99 bar, and k = 8 is no better than k = 4 (0.9758 vs 0.9761), so beyond the planted outliers the residual is spread across all channels: the per-channel A4 floor itself, not a few outliers. The 1.18–1.21x total-bytes edge over K8V4 does not come at acceptable error, and attention is 0.73–0.83x of K8V4 in numpy. K8V4 stays the recommended old-token cache; the 2-bit ternary weights carry the memory win (K8V4 total 1.75x below int8 + A8 at B = 8 / ctx 8192) either way. Caveat: synthetic K with 6 fixed hot channels per head and a static channel set.

## Links

- previous: [[2026-11-19_1455_kivi-per-channel-a4-k-sym-and-asym-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k4v4-k8v4-and-all-a8-u32-lut-total-bytes]]
- related: [[2026-11-19_1055_hadamard-rotated-k-before-a4-g32-k4v4-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-and-all-a8-u32-lut-total-bytes]], [[2026-11-19_0555_k8v4-a8-g64-keys-all-tokens-a4-g32-old-values-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k4v4-and-all-a8-u32-lut-total-bytes]]
- this wave's pair files: none. Quant Chief 18:05 assignment was pair_A no gap and pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-20T01:09:00Z; note "18:05 11/19: blobs unchanged; both no gap. JuniorHome tip by LLC bot 17:05 MT"). Roadmap blobs: `JuniorHome docs/ROADMAP.md` 4efb666, `JuniorLLM docs/BETA_TO_OS.md` bac6923.
- bench dir: `/workspace/bench1855_1119` (mixk.py, run1.txt, run2.txt, results_run1.json, results_run2.json)

## Script (`mixk.py`)

```python
#!/usr/bin/env python3
"""Mixed-precision old-token K (18:55 11/19). Derived from kivi4.py (14:55 11/19).
Per KV head, the top-k K channels by range (max-min) measured over the A8 recent window are kept at A8
(symmetric per-channel int8, f32 scale per (32-token block, channel)); the other HD-k channels use symmetric
per-channel A4 (KIVI sym, codes [-7,7], f32 scale per (block, channel)). Channels are permuted hot-first once per
head; at decode Q is permuted the same way (one gather per head per step), S = Qh Kh^T + Qc Kc^T.
V for old tokens stays per-token A4 g32. Last WIN=1024 tokens stay A8 g64.
Variants: mk4 / mk8 / mk16 (k hot channels). Controls: a8, k8v4 (05:55 rec), kc4v4 (14:55, = k 0).
Usage: OPENBLAS_NUM_THREADS=1 python3 mixk.py out.json"""
import json, sys, time, statistics
from concurrent.futures import ThreadPoolExecutor
import numpy as np
KVH, G, HD, WIN, W = 5, 4, 128, 1024, 8; SC = np.float32(1 / np.sqrt(HD)); NG = HD // 32
CTXS, BS, TKS = (2048, 8192), (1, 8), (1024, 2048)
FMTS = ("a8", "k8v4", "kc4v4", "mk4", "mk8", "mk16")
WB = {"u32": 16.65, "int8": 66.34}   # MiB, full bitnetd layer weights from 18:55 11/18 run1
ROUNDS, REPS = 7, 3
POOL = ThreadPoolExecutor(W)
rng = np.random.default_rng(11191855); TB_ = 32
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
def packw(c):  # (..., n) ints 0..15, n even -> (..., n/2) bytes
    n = c.shape[-1]; c = c.astype(np.uint8).reshape(*c.shape[:-1], n // 2, 2); return (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8)
def qcw(x, lim):   # per-channel symmetric over 32-token blocks: x (H,T,n) -> codes (H,T,n) int, scale (H,T/32,n)
    H, T, n = x.shape; g = x.reshape(H, T // TB_, TB_, n); m = np.abs(g).max(2) / float(lim) + 1e-12
    return np.clip(np.rint(g / m[:, :, None]), -lim, lim).astype(np.int16).reshape(H, T, n), m.astype(np.float32)
KS = {"mk4": 4, "mk8": 8, "mk16": 16}
class KV:
    def __init__(s, k, v):
        s.n = k.shape[1]; s.kf, s.vf = k, v
        (s.k8, s.ks8), (s.v8, s.vs8) = q8(k), q8(v)
        s.v4, s.v4s = q4(dq8(s.v8, s.vs8))
        kd = dq8(s.k8, s.ks8); s.kc4, s.kc4s = qc4(kd)
        win = kd[:, max(0, s.n - WIN):]; rngc = win.max(1) - win.min(1)          # (KVH, HD) channel range in A8 window
        s.mix = {}
        for f, kk in KS.items():
            perm = np.argsort(-rngc, axis=1, kind="stable")                         # hot first
            hot, cold = np.sort(perm[:, :kk], 1), np.sort(perm[:, kk:], 1); P = np.concatenate([hot, cold], 1)
            kp = np.take_along_axis(kd, P[:, None, :], 2)
            ch, sh = qcw(kp[:, :, :kk], 127); cc, scc = qcw(np.ascontiguousarray(kp[:, :, kk:]), 7)
            s.mix[f] = dict(P=P, kk=kk, ch=ch.astype(np.int8), sh=sh, cc=packw(cc + 8), sc=scc)
    def bytes(s, fmt):
        n8 = s.n if fmt == "a8" else min(s.n, WIN); n4 = s.n - n8; b8, b4 = HD + HD // 64 * 4, HD // 2 + NG * 4
        if fmt == "k8v4": return KVH * (s.n * b8 + n8 * b8 + n4 * b4)
        if fmt in KS:
            kk = KS[fmt]; kold = kk + kk * 4 // TB_ + (HD - kk) // 2 + (HD - kk) * 4 // TB_
            return KVH * (2 * n8 * b8 + n4 * b4 + n4 * kold + HD * 2)   # + int16 channel perm per head
        return 2 * KVH * (n8 * b8 + n4 * b4)
def online(m, l, acc, S, Vfn):
    mn = np.maximum(m, S.max(1, keepdims=True)); al = np.exp(m - mn); S -= mn; np.exp(S, out=S)
    l *= al; l += S.sum(1, keepdims=True); acc *= al; acc += Vfn(S); return mn
def vdq(kv, h, a, b):
    u = np.take(LUT16, kv.v4[h, a:b]).view(np.int8).reshape(b - a, NG, 32)
    return (u.astype(np.float32) * kv.v4s[h, a:b][:, :, None]).reshape(b - a, HD)
def tile_a8(kv, h, a, b, Q, st):
    K = dq8(kv.k8[h, a:b], kv.ks8[h, a:b]); V = dq8(kv.v8[h, a:b], kv.vs8[h, a:b])
    return online(*st, Q @ K.T, lambda P: P @ V)
def tile_kc4(kv, h, a, b, Q, st):
    T = b - a; a0, b0 = a // TB_, b // TB_
    u = np.take(LUT16, kv.kc4[h, a:b]).view(np.int8).reshape(T // TB_, TB_, HD)
    K = (u.astype(np.float32) * kv.kc4s[h, a0:b0][:, None]).reshape(T, HD); V = vdq(kv, h, a, b)
    return online(*st, Q @ K.T, lambda P: P @ V)
def tile_mix(kv, h, a, b, Qh, Qc, st, mx):
    T = b - a; a0, b0 = a // TB_, b // TB_; kk = mx["kk"]
    Kh = (mx["ch"][h, a:b].reshape(T // TB_, TB_, kk).astype(np.float32) * mx["sh"][h, a0:b0][:, None]).reshape(T, kk)
    u = np.take(LUT16, mx["cc"][h, a:b]).view(np.int8).reshape(T // TB_, TB_, HD - kk)
    Kc = (u.astype(np.float32) * mx["sc"][h, a0:b0][:, None]).reshape(T, HD - kk)
    S = Qc @ Kc.T; S += Qh @ Kh.T; V = vdq(kv, h, a, b)
    return online(*st, S, lambda P: P @ V)
def tile_k8v4(kv, h, a, b, Q, st):
    K = dq8(kv.k8[h, a:b], kv.ks8[h, a:b]); V = vdq(kv, h, a, b)
    return online(*st, Q @ K.T, lambda P: P @ V)
def head(kv, Q, h, fmt, TK):
    st = [np.full((G, 1), -np.inf, np.float32), np.zeros((G, 1), np.float32), np.zeros((G, HD), np.float32)]
    wb = 0 if fmt == "a8" else max(0, kv.n - WIN)
    if fmt in KS:
        mx = kv.mix[fmt]; Qp = Q[:, mx["P"][h]]; Qh = np.ascontiguousarray(Qp[:, :mx["kk"]]); Qc = np.ascontiguousarray(Qp[:, mx["kk"]:])
    for a in range(0, wb, TK):
        b = min(a + TK, wb)
        if fmt == "k8v4": st[0] = tile_k8v4(kv, h, a, b, Q, st)
        elif fmt == "kc4v4": st[0] = tile_kc4(kv, h, a, b, Q, st)
        elif fmt in KS: st[0] = tile_mix(kv, h, a, b, Qh, Qc, st, mx)
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
    print(f"{'cfg':28s} {'attn ms med (min-max)':>24s} {'KV MiB':>8s} {'min cos':>10s} {'max|d| vs k8v4':>15s}")
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
            for (fmt, TK), r in res.items(): r["maxdiff_vs_k8v4"] = float(np.abs(r["o"] - res[("k8v4", TK)]["o"]).max())
            for (fmt, TK), r in res.items():
                d = r["maxdiff_vs_k8v4"]; r.pop("o"); out["rows"].append(r)
                print(f"{r['key']:28s} {r['med']:8.2f} ({r['lo']:.2f}-{r['hi']:.2f})      {r['mib']:8.2f} {r['cos']:10.7f} {d:15.2e}")
            best = {f: min((res[(f, t)] for t in TKS), key=lambda r: r["med"]) for f in FMTS}
            for f in ("mk4", "mk8", "mk16", "kc4v4"):
                for refn in ("a8", "k8v4", "kc4v4"):
                    if f == refn: continue
                    k = f"{f}_vs_{refn}_ctx{n}_B{B}"; r = best[refn]["med"] / best[f]["med"]
                    out["compare"][k] = dict(speed=r, bytes=best[refn]["mib"] / best[f]["mib"], new=best[f]["key"], ref=best[refn]["key"])
                    print(f"  {k:30s} speed {r:5.2f}x  bytes {best[refn]['mib'] / best[f]['mib']:5.2f}x  ({best[f]['key']} vs {best[refn]['key']})")
            if B == 8:
                for wf in ("u32", "int8"):
                    for f in FMTS: out["compare"][f"total_{wf}_{f}_ctx{n}_B8"] = WB[wf] + best[f]["mib"]
                t = lambda wf, f: out["compare"][f"total_{wf}_{f}_ctx{n}_B8"]
                for f in FMTS:
                    print(f"  total ctx{n} B8 u32+{f}: {t('u32', f):7.2f} MiB | vs int8+a8 {t('int8','a8')/t('u32', f):5.2f}x | vs int8+same {t('int8', f)/t('u32', f):5.2f}x | vs u32+k8v4 {t('u32','k8v4')/t('u32', f):5.2f}x")
    # hot-channel hit rate vs the 6 planted hot channels is not observable here (kvhist hides them); report selection cost
    for B in BS:
        win = rng.standard_normal((B * KVH, WIN, HD)).astype(np.float32)
        ts = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            for _ in range(20): r = win.max(1) - win.min(1); np.argsort(-r, axis=1, kind="stable")
            ts.append((time.perf_counter() - t0) / 20 * 1e3)
        out["compare"][f"select_B{B}"] = dict(med=statistics.median(ts), lo=min(ts), hi=max(ts))
        print(f"  hot-channel selection (once per block of 32 steps if refreshed) B{B}: {statistics.median(ts):.4f} ms ({min(ts):.4f}-{max(ts):.4f})")
        q = rng.standard_normal((B * KVH, G, HD)).astype(np.float32); P = np.argsort(rng.standard_normal((B * KVH, HD)), 1)
        ts = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            for _ in range(200):
                for i in range(B * KVH): Qp = q[i][:, P[i]]; np.ascontiguousarray(Qp[:, :8]); np.ascontiguousarray(Qp[:, 8:])
            ts.append((time.perf_counter() - t0) / 200 * 1e3)
        out["compare"][f"qgather_B{B}"] = dict(med=statistics.median(ts), lo=min(ts), hi=max(ts))
        print(f"  Q permute+split per step B{B}: {statistics.median(ts):.4f} ms ({min(ts):.4f}-{max(ts):.4f})")
    json.dump(out, open(sys.argv[1], "w"), indent=1)
main()
```

## Next question

Three A4 K fixes (rotation 0.906, per-channel 0.938–0.957, per-channel + top-16 A8 0.982) all fall short of K8V4 at B = 8 / ctx 8192, so the floor is the 4-bit K width itself. Is there a middle K width that holds K8V4 accuracy for fewer bytes? Test per-channel A6 K (6-bit codes bit-packed 4 values per 3 bytes, f32 scale per 32-token block) and A5 K for old tokens with V at A4 g32: does either reach min cos ≥ K8V4 − 0.002 at ≥ 1.1x fewer total bytes than K8V4 at B = 8 / ctx 8192, and what does the 6-bit unpack cost per decode step in numpy? Also split the K8V4 error into K-only and V-only parts to see whether V A4 is now the limit.


close: sha256(prev || this)[:16] = 0b5f67ce96e8133e
