---
local: true
date: 2026-11-20
wave: "2026-11-20 05:55 (ran ~05:50–06:10 MT; Chief last_run 2026-11-20T12:10:00Z = 05:10 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-stream, kv-cache, hadamard, rotation, a4-v, k8v4, kc6, kc5, a8-g64, recent-window, gqa, numpy, not-viable]
---

# Rotating V before A4 g32 is nearly free but buys little. At B = 8 / ctx 8192, a per-head Hadamard rotation of old-token V lifts K8V4's min cos only from **0.99138** to **0.99222** (all-A8 is 0.99756), at the same **86.34 MiB** total and 0.99–1.12x of K8V4's attention speed. With per-channel A6 K it reaches **0.99081** (inside the K8V4 − 0.002 bar) but still saves only **1.08x** total bytes vs K8V4. With A5 K it stays at **0.98591**, 0.0055 below K8V4. No new 1.5x, and no layout that beats K8V4 on both bars. Not viable. Keep K8V4. Rotation is a cheap optional accuracy bump, not a byte win.

## Question

[[2026-11-20_0255_per-channel-a5-a6-k-old-kv-plus-a4-v-and-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-k4v4-and-all-a8-u32-lut-total-bytes]] found K8V4's whole gap to all-A8 comes from the A4 V cache. Its next question:
- Does Hadamard-rotated A4 g32 V (with K8, kc6 or kc5 K) lift K8V4's min cos toward all-A8 (≥ 0.995)?
- Does kc5 K + rotated A4 V reach ≥ K8V4 − 0.002 at the 1.15x total-byte saving vs K8V4?
- What are the write-side rotation cost per new token and the per-step un-rotate cost at B = 8?

## Method

- Same attention-only microbench and synthetic K/V generator as 05:55 11/19 – 02:55 11/20 (`a6k.py` → `hv4.py`). Gaussian K/V with a log-normal per-channel spread (K σ 0.5 plus 6 planted 6x hot channels per head, V σ 0.3), slow positional drift on K. New seed 11200555.
- Shapes: B ∈ {1, 8} streams × 5 KV heads × 4 q heads (GQA 20/5), HD 128, ctx ∈ {2048, 8192}. The last 1024 tokens stay at A8 g64 for K and V.
- Rotation: HR = diag(±1) · H128 / √128 (Sylvester Hadamard, random signs, orthonormal to 1e-5). Old V is quantized per token as A4 g32 from (A8-dequantized V) · HR. Since P · (V HR) = (P · V) HR, old tiles accumulate in the rotated domain. All old tiles precede the window, so the accumulator is un-rotated once per (stream, head) per step (acc · HRᵀ, 4 × 128 × 128). Nothing extra is done per token at read time, and no bytes are added per token (HR is one shared 64 KiB f32 matrix).
- Variants: `a8`, `k8v4` (current rec), `k8hv4`, `kc6v4`, `kc6hv4`, `kc5v4`, `kc5hv4` (kc6 / kc5 = per-channel A6 / A5 K with an f32 scale per (32-token block, channel), as 02:55 11/20).
- Attention: online softmax per (stream, KV head) on a W = 8 thread pool, `OPENBLAS_NUM_THREADS=1`, TK ∈ {1024, 2048}, best TK per variant, 7 rounds × 3 reps, median (min–max).
- Side costs (one thread, 40 stream-heads at B = 8, 7 rounds): write-side A4 g32 quant of one new aged-out V token per head with and without rotation, and the per-step un-rotate (batched matmul and per-head loop).
- Total bytes = full bitnetd layer weights from 18:55 11/18 run1 (u32 LUT 16.65 MiB, int8 66.34 MiB; not re-timed) + KV measured at B = 8.
- Error: min over (stream, head) of cos(attention output, exact f32-KV softmax). Two independent runs gave identical error; timing differs.

## Command

```
cd /workspace/bench0555_1120
OPENBLAS_NUM_THREADS=1 python3 hv4.py results_run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 python3 hv4.py results_run2.json > run2.txt
```

## Results: attention ms per decode step, all streams (run1 / run2, best TK, median)

| ctx / B | a8 | k8v4 | k8hv4 | kc6v4 | kc6hv4 | kc5v4 | kc5hv4 |
|---|---|---|---|---|---|---|---|
| 2048 / 1 | 1.11 / 0.87 | 1.34 / 1.22 | 1.33 / 1.18 | 1.76 / 1.48 | 1.86 / 1.62 | 1.39 / 1.40 | 1.49 / 1.45 |
| 2048 / 8 | 6.45 / 5.61 | 9.57 / 8.82 | 10.23 / 9.08 | 12.87 / 12.36 | 12.81 / 12.23 | 10.77 / 11.20 | 10.53 / 10.56 |
| 8192 / 1 | 3.19 / 3.08 | 4.89 / 3.76 | 3.73 / 4.02 | 5.16 / 5.24 | 5.52 / 5.49 | 4.86 / 4.76 | 5.89 / 4.99 |
| 8192 / 8 | 21.05 / 18.45 | 28.44 / 24.28 | 25.33 / 24.46 | 37.81 / 38.64 | 37.55 / 38.25 | 31.90 / 33.03 | 32.46 / 31.86 |

Spread at B = 8 / ctx 8192 (TK 2048, ms): a8 18.09–21.65 | 18.04–21.77; k8v4 23.45–32.93 | 23.75–25.45; k8hv4 24.65–26.21 | 24.19–25.13; kc6hv4 36.88–40.07 | 37.56–39.07; kc5hv4 32.15–33.48 | 30.71–32.21.

## Results: min cos vs exact f32 attention (identical in both runs)

| ctx / B | a8 | k8v4 | k8hv4 | kc6v4 | kc6hv4 | kc5v4 | kc5hv4 |
|---|---|---|---|---|---|---|---|
| 2048 / 1 | 0.99668 | 0.99449 | 0.99554 | 0.99242 | 0.99367 | 0.99168 | 0.99286 |
| 2048 / 8 | 0.99668 | 0.99177 | 0.99399 | 0.99178 | 0.99367 | 0.99066 | 0.99149 |
| 8192 / 1 | 0.99776 | 0.99138 | 0.99222 | 0.98888 | 0.99081 | 0.98756 | 0.98724 |
| 8192 / 8 | 0.99756 | 0.99138 | 0.99222 | 0.98888 | 0.99081 | 0.98527 | 0.98591 |

Old-token V relative Frobenius error vs f32 V (median over stream-heads): A4 g32 0.1111 → rotated A4 g32 0.0970 (ctx 2048 and 8192 alike).

## Results: bytes at B = 8

| variant | KV MiB ctx 2048 | KV MiB ctx 8192 | total ctx 8192 (u32 + KV) | vs int8 + A8 (151.34 MiB) | vs u32 + K8V4 |
|---|---|---|---|---|---|
| a8 | 21.25 | 85.00 | 101.65 | 1.49x | 0.85x |
| k8v4 | 19.06 | 69.69 | 86.34 | 1.75x | 1.00x |
| k8hv4 | 19.06 | 69.69 | 86.34 | 1.75x | 1.00x |
| kc6v4 / kc6hv4 | 18.12 | 63.12 | 79.78 | 1.90x | 1.08x |
| kc5v4 / kc5hv4 | 17.50 | 58.75 | 75.40 | 2.01x | 1.15x |

("vs int8 + A8" = int8 weights 66.34 MiB + all-A8 KV 85.00 MiB = 151.34 MiB, divided by the row's total.)

Side costs, B = 8 / 40 stream-heads, µs (run1 | run2): write A4 g32 28.81 | 28.08; write rotate + A4 g32 38.20 | 37.44 (+9.4 µs per new token, all streams); un-rotate per step batched 23.13 | 23.06, per-head loop 58.18 | 58.11. Against a 24–25 ms attention step, the un-rotate is ~0.1–0.2% and the write is noise.

## Verdict

**not-viable.** Rotation works as algebra (no per-token read cost, +9 µs write, +23 µs un-rotate per B = 8 step) but the synthetic V has no channel outliers, so it only lowers old-V error from 0.111 to 0.097:
- K8 + rotated V: +0.0008 min cos at B = 8 / ctx 8192 (0.99222), still 0.0053 below all-A8, same bytes. Not the ≥ 0.995 target.
- kc6 K + rotated V: 0.99081, inside the K8V4 − 0.002 bar (0.98938), but 1.08x total bytes vs K8V4 and 0.63–0.76x attention speed. Below the 1.1x byte bar.
- kc5 K + rotated V: 0.98591, still 0.0055 below K8V4. The 1.15x saving is not unlocked.

K8V4 stays the recommended old-token cache; rotated V (k8hv4) is an optional, near-free accuracy bump. The real memory win is still the 2-bit ternary weights: K8V4 total is 1.75x below int8 + A8 at B = 8 / ctx 8192 (already recorded 05:55 11/19, not new).

Caveat: synthetic Gaussian V. Real V caches with heavy outlier channels would gain more from rotation; this bench can't show that.

## Links

- previous: [[2026-11-20_0255_per-channel-a5-a6-k-old-kv-plus-a4-v-and-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-k4v4-and-all-a8-u32-lut-total-bytes]]
- related: [[2026-11-19_1055_hadamard-rotated-k-before-a4-g32-k4v4-old-kv-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k8v4-and-all-a8-u32-lut-total-bytes]] (same rotation on K)
- this wave's pair files: none.
  - Quant Chief 05:05 assignment was pair_A no gap and pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-20T12:10:00Z).
  - Chief note: "05:05 11/20: blobs unchanged; both no gap." LLC skip path: `JuniorHome/scripts/advance_gguf_operator_sets_named_20261120.py` (not touched).
  - Roadmap blobs: `JuniorHome docs/ROADMAP.md` 4efb666, `JuniorLLM docs/BETA_TO_OS.md` bac6923.
- bench dir: `/workspace/bench0555_1120` (hv4.py, run1.txt, run2.txt, results_run1.json, results_run2.json)

## Script (`hv4.py`)

```python
#!/usr/bin/env python3
"""Hadamard-rotated A4 g32 old-token V (05:55 11/20). Derived from a6k.py (02:55 11/20).
Old-token V (aged out of the A8 g64 window) is rotated per head by HR = diag(+-1) @ H128 / sqrt(128) (Sylvester Hadamard,
random signs) before per-token A4 g32. Since P @ (V HR) = (P @ V) HR, attention accumulates old tiles in the rotated
domain and un-rotates the accumulator once per (stream, head) per step: acc @ HR.T (G x HD x HD), nothing per token.
Old-token K: k8 (A8 g64), kc6 / kc5 (per-channel A6 / A5, f32 scale per (32-token block, channel), as a6k.py).
Variants: a8, k8v4, k8hv4, kc6v4, kc6hv4, kc5v4, kc5hv4. Last WIN=1024 tokens stay A8 g64 for K and V.
Usage: OPENBLAS_NUM_THREADS=1 python3 hv4.py out.json"""
import json, sys, time, statistics
from concurrent.futures import ThreadPoolExecutor
import numpy as np
KVH, G, HD, WIN, W = 5, 4, 128, 1024, 8; SC = np.float32(1 / np.sqrt(HD)); NG = HD // 32; TB_ = 32
CTXS, BS, TKS = (2048, 8192), (1, 8), (1024, 2048)
FMTS = ("a8", "k8v4", "k8hv4", "kc6v4", "kc6hv4", "kc5v4", "kc5hv4")
WB = {"u32": 16.65, "int8": 66.34}   # MiB, full bitnetd layer weights from 18:55 11/18 run1 (not re-timed)
ROUNDS, REPS = 7, 3
POOL = ThreadPoolExecutor(W)
rng = np.random.default_rng(11200555)
H = np.array([[1.0]]);
while H.shape[0] < HD: H = np.block([[H, H], [H, -H]])
HR = (rng.choice([-1.0, 1.0], HD)[:, None] * H / np.sqrt(HD)).astype(np.float32); HRT = np.ascontiguousarray(HR.T)
assert np.abs(HR @ HRT - np.eye(HD)).max() < 1e-5
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
def dq4(p, s):
    u = np.take(LUT16, p).view(np.int8).reshape(*p.shape[:-1], NG, 32)
    return (u.astype(np.float32) * s[..., None]).reshape(*p.shape[:-1], HD)
def pack4(c):
    n = c.shape[-1]; c = c.astype(np.uint8).reshape(*c.shape[:-1], n // 2, 2); return (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8)
def qcw(x, lim):
    Hh, T, n = x.shape; g = x.reshape(Hh, T // TB_, TB_, n); m = np.abs(g).max(2) / float(lim) + 1e-12
    return np.clip(np.rint(g / m[:, :, None]), -lim, lim).astype(np.int16).reshape(Hh, T, n), m.astype(np.float32)
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
def unpack5(lo, hb):
    u = np.take(LUTU16, lo).view(np.uint8).reshape(lo.shape[0], HD)
    u |= np.unpackbits(hb, axis=-1, bitorder="little") << 4; u -= 16; return u.view(np.int8)
def kvhist(n):  # same generator as 05:55 11/19 - 02:55 11/20
    ch = np.exp(rng.normal(0, 0.5, (KVH, 1, HD))).astype(np.float32); ch[:, :, rng.choice(HD, 6, replace=False)] *= 6
    k = (rng.standard_normal((KVH, n, HD)).astype(np.float32) + 0.3 * np.sin(np.arange(n) / 300.0)[None, :, None]) * ch
    v = rng.standard_normal((KVH, n, HD)).astype(np.float32) * np.exp(rng.normal(0, 0.3, (KVH, 1, HD))).astype(np.float32)
    return k.astype(np.float32), v.astype(np.float32)
KB = {"kc5": HD // 2 + HD // 8, "kc6": 3 * HD // 4}
def kpart(f): return "k8v" if f.startswith("k8") else f[:3]
class KV:
    def __init__(s, k, v):
        s.n = k.shape[1]; s.kf, s.vf = k, v
        (s.k8, s.ks8), (s.v8, s.vs8) = q8(k), q8(v)
        vd = dq8(s.v8, s.vs8); s.v4, s.v4s = q4(vd); s.vh4, s.vh4s = q4(vd @ HR); kd = dq8(s.k8, s.ks8)
        c, s.kc5s = qcw(kd, 15); c = (c + 16).astype(np.uint8); s.kc5lo = pack4(c & 15)
        s.kc5hb = np.packbits((c >> 4).astype(np.uint8), axis=-1, bitorder="little")
        c, s.kc6s = qcw(kd, 31); s.kc6 = pack6(c + 32)
    def bytes(s, fmt):
        n8 = s.n if fmt == "a8" else min(s.n, WIN); n4 = s.n - n8; b8, b4 = HD + HD // 64 * 4, HD // 2 + NG * 4
        if fmt == "a8": return 2 * KVH * s.n * b8
        kold = b8 if kpart(fmt) == "k8v" else KB[kpart(fmt)] + HD * 4 // TB_
        return KVH * (2 * n8 * b8 + n4 * (kold + b4))   # rotation adds no per-token bytes (HR is one shared 64 KiB f32 matrix)
def online(m, l, acc, S, Vfn):
    mn = np.maximum(m, S.max(1, keepdims=True)); al = np.exp(m - mn); S -= mn; np.exp(S, out=S)
    l *= al; l += S.sum(1, keepdims=True); acc *= al; acc += Vfn(S); return mn
def vdq(kv, h, a, b, fmt):
    if "hv4" in fmt: return dq4(kv.vh4[h, a:b], kv.vh4s[h, a:b])   # rotated domain
    return dq4(kv.v4[h, a:b], kv.v4s[h, a:b])
def kdq(kv, h, a, b, fmt):
    T = b - a; a0, b0 = a // TB_, b // TB_; kf = kpart(fmt)
    if kf == "k8v": return dq8(kv.k8[h, a:b], kv.ks8[h, a:b])
    if kf == "kc5": u, sc = unpack5(kv.kc5lo[h, a:b], kv.kc5hb[h, a:b]), kv.kc5s
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
    if "hv4" in fmt and wb > 0: st[2] = st[2] @ HRT   # old tiles all precede the window: one un-rotate per head per step
    for a in range(wb, kv.n, TK if fmt == "a8" else WIN):
        st[0] = tile_a8(kv, h, a, min(a + (TK if fmt == "a8" else WIN), kv.n), Q, st)
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
    print(f"{'cfg':28s} {'attn ms med (min-max)':>24s} {'KV MiB':>8s} {'min cos':>10s}")
    for n in CTXS:
        kvs8 = [KV(*kvhist(n)) for _ in range(max(BS))]
        q8all = (rng.standard_normal((max(BS), KVH, G, HD)).astype(np.float32) * SC)
        wb = n - WIN
        for nm, fn in (("v4", lambda kv, h: dq4(kv.v4[h, :wb], kv.v4s[h, :wb])), ("hv4", lambda kv, h: dq4(kv.vh4[h, :wb], kv.vh4s[h, :wb]) @ HRT)):
            e = [np.linalg.norm(fn(kv, h) - kv.vf[h, :wb]) / np.linalg.norm(kv.vf[h, :wb]) for kv in kvs8 for h in range(KVH)]
            out["compare"][f"Verr_{nm}_ctx{n}"] = float(np.median(e)); print(f"  old-V rel err {nm} ctx{n}: median {np.median(e):.5f} max {max(e):.5f}")
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
            for f in ("k8hv4", "kc6v4", "kc6hv4", "kc5v4", "kc5hv4"):
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
    # side costs at B = 8 (40 stream-heads), single thread, microseconds per decode step
    X = rng.standard_normal((8 * KVH, HD)).astype(np.float32); A = rng.standard_normal((8 * KVH, G, HD)).astype(np.float32)
    out["side"]["write_q4"] = tmed(lambda: q4(X), 200)
    out["side"]["write_rot_q4"] = tmed(lambda: q4(X @ HR), 200)
    out["side"]["unrot_batched"] = tmed(lambda: A @ HRT, 200)
    out["side"]["unrot_per_head"] = tmed(lambda: [A[i] @ HRT for i in range(8 * KVH)], 50)
    for k, v in out["side"].items(): print(f"  side {k:16s} B8 (40 stream-heads): {v['med']:8.2f} us ({v['lo']:.2f}-{v['hi']:.2f})")
    json.dump(out, open(sys.argv[1], "w"), indent=1)
main()
```

## Next question

Rotation can't fix error that is just 4-bit Gaussian rounding. The remaining V lever is width or group size. With K8 K at B = 8 / ctx 8192: does A5 g32 old V (nibble plane + 1-bit plane, +16 B/token), A4 g16 V (+16 B/token of scales), or a 3-tier recency cache (A8 last 1024 → A6 V next 2048 → A4 V oldest) lift min cos to ≥ 0.995 at ≤ 1.05x the K8V4 bytes, and what does each cost in attention speed?

close: sha256(prev || this)[:16] = b294867756d359d7
