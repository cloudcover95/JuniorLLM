---
local: true
date: 2026-11-19
wave: "2026-11-19 10:55 (ran ~10:40–11:00 MT; Chief last_run 2026-11-19T17:10:00Z ~10:10 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-stream, kv-cache, hadamard, rotation, k4v4, k8v4, a4-g32, a8-g64, recent-window, gqa, numpy, not-viable]
---

# A fixed per-head Hadamard rotation of K before A4 g32 lifts K4V4 min attention-output cos from **0.579 to 0.906** at B = 8 / ctx 8192 (0.809 → 0.934 at B = 1) but does **not** reach the 0.99 bar that K8V4 clears (0.991). Bytes stay at the K4V4 level (71.03 MiB total with 2-bit weights, **2.13x below int8 + A8**), attention runs at 0.97–1.11x of unrotated K4V4 and 0.86–0.93x of K8V4. Not viable: the extra 1.23x of bytes over K8V4 comes with unacceptable error. Keep K8V4.

## Question

[[2026-11-19_0555_k8v4-a8-g64-keys-all-tokens-a4-g32-old-values-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k4v4-and-all-a8-u32-lut-total-bytes]] showed that K8V4 fixes the hot-K-channel accuracy loss of K4V4 at a cost of 2.13x → 1.75x total-bytes edge. Its next question: does a fixed per-head Hadamard rotation of K before A4 g32 (folded into Q at decode, so scores are unchanged in exact arithmetic) bring K4V4 min cos back to ≥ 0.99, recovering the 2.13x edge at B = 8 / ctx 8192, and what does the rotation cost per decode step?

## Method

- Same attention-only microbench and synthetic K/V as 05:55 (`k8v4.py` → `hadk4.py`): Gaussian K/V with log-normal per-channel spread, 6 hot K channels × 6, slow positional drift; new seed 11191055. B ∈ {1, 8} streams × 5 KV heads × 4 q heads (GQA 20/5), HD 128, ctx ∈ {2048, 8192}, last 1024 tokens at A8 g64 (unrotated). Online softmax per (stream, KV head) on a W = 8 thread pool, `OPENBLAS_NUM_THREADS=1`, TK ∈ {1024, 2048}, best TK per variant used.
- New variant `hk4v4`: R_h = H128 · diag(random ±1) / √128 per KV head (orthonormal Sylvester Hadamard). Aged-out K rows are rotated (K R) then A4 g32; V A4 g32 as K4V4. At decode Q R is computed once per (stream, KV head) **inside the timed step** and scores the old tiles; window tiles use Q. f32 exactness check: max |Q Kᵀ − (Q R)(K R)ᵀ| = 1.14e-05.
- Controls: `a8` (all A8 g64), `a4deq` (unrotated K4V4), `k8v4` (05:55 recommendation).
- Write side: per step each stream ages one K row per KV head out of the window (dequant A8 → [rotate] → A4 g32); timed separately, 200 iterations × 7 rounds.
- Total bytes = full bitnetd layer weights from 18:55 11/18 run1 (u32 LUT 16.65 MiB, int8 66.34 MiB; not re-timed) + measured KV at B = 8.
- Error: min over (stream, head) of cos(attention output, exact f32-KV softmax). 7 rounds × 3 reps per config, median (min–max), two independent runs (same data, timing differs).

## Command

```
cd /workspace/bench1055_1119
OPENBLAS_NUM_THREADS=1 python3 hadk4.py results_run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 python3 hadk4.py results_run2.json > run2.txt
```

## Results: attention ms per decode step, all streams (run1 / run2, best TK)

| ctx / B | a8 | a4deq (K4V4) | k8v4 | hk4v4 | KV MiB a8 / K4V4 = hK4V4 / K8V4 | hk4v4 vs a4deq | hk4v4 vs k8v4 | hk4v4 vs a8 |
|---|---|---|---|---|---|---|---|---|
| 2048 / 1 | 1.22 / 1.10 | 2.09 / 1.52 | 2.03 / 1.51 | 1.90 / 1.37 | 2.66 / 2.11 / 2.38 | 1.10x / 1.11x | 1.07x / 1.10x | 0.64x / 0.80x |
| 2048 / 8 | 8.79 / 7.17 | 13.84 / 11.57 | 13.23 / 10.90 | 13.28 / 11.67 | 21.25 / 16.88 / 19.06 | 1.04x / 0.99x | 1.00x / 0.93x | 0.66x / 0.61x |
| 8192 / 1 | 3.97 / 3.84 | 5.81 / 6.02 | 4.76 / 4.77 | 5.32 / 5.42 | 10.62 / 6.80 / 8.71 | 1.09x / 1.11x | 0.89x / 0.88x | 0.75x / 0.71x |
| 8192 / 8 | 25.11 / 26.98 | 32.97 / 36.39 | 29.08 / 32.48 | 33.97 / 34.96 | 85.00 / 54.38 / 69.69 | 0.97x / 1.04x | 0.86x / 0.93x | 0.74x / 0.77x |

Spread run1 at B = 8 / ctx 8192 (best TK): a8 24.71–25.87, a4deq 28.48–35.24, k8v4 25.45–32.14, hk4v4 31.53–41.33 ms.

Error (min attention-output cos; identical in both runs):

| ctx / B | a8 | a4deq (K4V4) | k8v4 | hk4v4 |
|---|---|---|---|---|
| 2048 / 1 | 0.99837 | 0.88073 | 0.99479 | 0.97000 |
| 2048 / 8 | 0.99518 | 0.83747 | 0.99021 | 0.94946 |
| 8192 / 1 | 0.99781 | 0.80941 | 0.99231 | 0.93414 |
| 8192 / 8 | 0.99640 | 0.57901 | 0.99086 | 0.90572 |

Write-side age-out cost per decode step (run1 / run2 median): B = 1 plain 0.0165 / 0.0161 ms, rotated 0.0276 / 0.0277 ms; B = 8 plain 0.0342 / 0.0338 ms, rotated 0.1098 / 0.1095 ms (+0.076 ms, ~0.2–0.3% of a B = 8 / ctx 8192 attention step).

Total bytes per layer at B = 8 (weights + KV, MiB):

| ctx | u32 + a8 | u32 + K4V4 = u32 + hK4V4 | u32 + K8V4 | int8 + a8 | u32+hK4V4 vs int8+a8 | u32+K8V4 vs int8+a8 |
|---|---|---|---|---|---|---|
| 2048 | 37.90 | 33.52 | 35.71 | 87.59 | 2.61x | 2.45x |
| 8192 | 101.65 | 71.03 | 86.34 | 151.34 | 2.13x | 1.75x |

## Verdict

**not-viable.** The rotation is exact (1e-5 in f32), cheap (Q R per head on the read side is inside the measured step and leaves hk4v4 within ±11% of unrotated K4V4; write side +0.08 ms at B = 8) and it clearly spreads the hot channels (min cos +0.09 to +0.33). But it stops at 0.906–0.970, below the 0.99 accuracy that K8V4 and all-A8 keep, and the gap widens with context and stream count. The 2.13x total-bytes edge it would recover over int8 + A8 (vs 1.75x for K8V4) is not worth a 0.906 attention cos. K8V4 stays the recommended old-token cache; the ternary weights still carry the memory win either way.

## Links

- previous: [[2026-11-19_0555_k8v4-a8-g64-keys-all-tokens-a4-g32-old-values-plus-a8-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-vs-k4v4-and-all-a8-u32-lut-total-bytes]]
- related: [[2026-11-19_0255_fused-a4-g32-nibble-attention-scale-folded-and-byte-lut-scoring-vs-f32-dequant-multi-stream-decode-b1-b8-ctx-2048-8192]], [[2026-11-18_1855_a4-g32-nibble-old-kv-plus-a8-g64-recent-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-u32-lut-vs-int8-weights-total-bytes-full-bitnetd-layer]]
- this wave's pair files: none. Quant Chief 10:05 assignment was pair_A no gap and pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-19T17:10:00Z). Roadmap blobs: `JuniorHome docs/ROADMAP.md` 4efb666, `JuniorLLM docs/BETA_TO_OS.md` bac6923.
- bench dir: `/workspace/bench1055_1119` (hadk4.py, run1.txt, run2.txt, results_run1.json, results_run2.json)

## Script (`hadk4.py`)

```python
#!/usr/bin/env python3
"""Hadamard-rotated K4V4 (10:55 11/19): before A4 g32 quant of an aged-out K row, rotate it by a fixed per-head
orthonormal R = H128 diag(sign) / sqrt(128) (Sylvester Hadamard x random signs). At decode Q is rotated once per
(stream, KV head) by the same R (Q R . K R = Q . K exactly), and the rotated Q scores the old A4 tiles; the A8 window is
unrotated. Derived from k8v4.py (05:55 11/19).
New variant: hk4v4 = rotated A4 g32 K + A4 g32 V for old tokens (K4V4 bytes); rotation cost (Q R per head) is inside the
timed step. Write-side cost (rotate+quantize one aged-out K row per stream per step) is timed separately. Attention-only microbench of the
18:55 11/18 multi-stream decode KV layout: B streams x 5 KV heads x 4 q heads (GQA 20/5), HD 128, ctx n, last WIN=1024
tokens at A8 g64, older tokens at A4 g32 (two 4-bit codes/byte, symmetric [-7,7], f32 scale per 32 dims).
Old-region variants (window tile identical in all: A8 dequant):
  a8      : whole context A8 g64 dequant (18:55 control, 1.56x more KV bytes)
  a4deq   : 18:55 path: u16 LUT -> int8 codes -> f32 * scale -> Q K^T, P V   (baseline to beat)
  a4fold  : f32 pair LUT (byte -> 2 f32 codes, no int8 hop), scales folded: per-group partial scores
            S_g = Qbd @ C^T (block-diag Q, 16 x 128), S = sum_g S_g * ks[:,g]; V: P*vs per group then (P_g) @ C_g
  a4byte  : T-MAC style byte-LUT scoring: table[pos,byte,q] = q . dec(byte) for 64 byte positions; S_g = sum of 16 gathers;
            V via the f32 pair LUT + folded scales (same as a4fold)
Online softmax per (stream, KV head) over TK-key tiles on a W-thread pool. Error: cos of attention output vs f32-KV
exact softmax. Usage: OPENBLAS_NUM_THREADS=1 python3 k8v4.py out.json"""
import json, sys, time, statistics
from concurrent.futures import ThreadPoolExecutor
import numpy as np
KVH, G, HD, WIN, W = 5, 4, 128, 1024, 8; SC = np.float32(1 / np.sqrt(HD)); NG = HD // 32
CTXS, BS, TKS = (2048, 8192), (1, 8), (1024, 2048)
FMTS = ("a8", "a4deq", "k8v4", "hk4v4")
WB = {"u32": 16.65, "int8": 66.34}   # MiB, full bitnetd layer weights from 18:55 11/18 run1
ROUNDS, REPS = 7, 3
POOL = ThreadPoolExecutor(W)
rng = np.random.default_rng(11191055)
def hadamard(n):
    H = np.ones((1, 1), np.float32)
    while H.shape[0] < n: H = np.block([[H, H], [H, -H]])
    return H
ROT = np.stack([hadamard(HD) * rng.choice(np.float32([-1, 1]), HD)[None, :] / np.sqrt(HD) for _ in range(KVH)]).astype(np.float32)
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
        s.hk4, s.hk4s = q4(np.einsum("hnd,hde->hne", dq8(s.k8, s.ks8), ROT))          # rotated, then A4 g32
    def bytes(s, fmt):
        n8 = s.n if fmt == "a8" else min(s.n, WIN); n4 = s.n - n8; b8, b4 = HD + HD // 64 * 4, HD // 2 + NG * 4
        if fmt == "k8v4": return KVH * (s.n * b8 + n8 * b8 + n4 * b4)
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
def tile_hk4v4(kv, h, a, b, Qr, st):
    def dq4(c, sc):
        u = np.take(LUT16, c).view(np.int8).reshape(b - a, NG, 32)
        return (u.astype(np.float32) * sc[:, :, None]).reshape(b - a, HD)
    K = dq4(kv.hk4[h, a:b], kv.hk4s[h, a:b]); V = dq4(kv.v4[h, a:b], kv.v4s[h, a:b])
    return online(*st, Qr @ K.T, lambda P: P @ V)
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
    if fmt == "hk4v4": Qr = Q @ ROT[h]      # per-step rotation cost, inside the timed path
    for a in range(0, wb, TK):
        b = min(a + TK, wb)
        if fmt == "a4deq": st[0] = tile_a4deq(kv, h, a, b, Q, st)
        elif fmt == "k8v4": st[0] = tile_k8v4(kv, h, a, b, Q, st)
        elif fmt == "hk4v4": st[0] = tile_hk4v4(kv, h, a, b, Qr, st)
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
            for f in ("hk4v4", "k8v4", "a4deq"):
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
    # write-side: each step, per stream, one K row per KV head ages out of the window: dequant A8 -> (rotate) -> A4 g32
    for B in BS:
        kr = rng.standard_normal((B, KVH, 1, HD)).astype(np.float32); c8, s8 = q8(kr)
        for name, rot in (("plain", False), ("rot", True)):
            ts = []
            for _ in range(ROUNDS):
                t0 = time.perf_counter()
                for _ in range(200):
                    x = dq8(c8, s8)
                    if rot: x = np.einsum("bhnd,hde->bhne", x, ROT)
                    q4(x)
                ts.append((time.perf_counter() - t0) / 200 * 1e3)
            out["compare"][f"write_{name}_B{B}"] = dict(med=statistics.median(ts), lo=min(ts), hi=max(ts))
            print(f"  write-side age-out B{B} {name}: {statistics.median(ts):.4f} ms ({min(ts):.4f}-{max(ts):.4f})")
    # exactness of the rotation (f32): max |Q K^T - (QR)(KR)^T|
    k = rng.standard_normal((KVH, 64, HD)).astype(np.float32); qq = rng.standard_normal((KVH, G, HD)).astype(np.float32)
    d = max(float(np.abs(qq[h] @ k[h].T - (qq[h] @ ROT[h]) @ (k[h] @ ROT[h]).T).max()) for h in range(KVH))
    out["compare"]["rotation_exactness_maxabs"] = d; print(f"  rotation exactness max|dS| f32: {d:.2e}")
    json.dump(out, open(sys.argv[1], "w"), indent=1)
main()
```

## Next question

Rotation alone cannot fix K4 because per-token groups still share one scale across channels whose ranges differ by an order of magnitude. Does KIVI-style per-channel K quantization at A4 (scale per channel over 32- or 64-token blocks, V stays per-token A4 g32), with or without the Hadamard rotation, reach min cos ≥ 0.99 at ≤ K8V4 bytes, and what does the per-channel scale layout cost in numpy decode attention at B = 8 / ctx 8192?

close: sha256(prev || this)[:16] = 1f60eb12f58fe44e
