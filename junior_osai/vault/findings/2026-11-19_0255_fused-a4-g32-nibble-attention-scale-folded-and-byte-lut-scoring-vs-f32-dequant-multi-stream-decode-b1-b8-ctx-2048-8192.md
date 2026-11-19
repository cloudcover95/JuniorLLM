---
local: true
date: 2026-11-19
wave: "2026-11-19 02:55 (ran ~02:47–03:00 MT; Chief last_run 2026-11-19T09:12:00Z ~02:12 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-stream, kv-cache, a4-g32, a8-g64, recent-window, fused-dequant, scale-folding, byte-lut, t-mac-style, gqa, numpy, not-viable]
---

# Skipping the f32 scale-dequant for A4 g32 old-token attention does not make it faster in numpy: the scale-folded path is **0.78–0.98x** and the T-MAC-style byte-LUT scoring path **0.38–0.70x** of the 18:55 dequant path, and at B = 8 / ctx 8192 the best A4 path stays **0.65–0.71x of all-A8 speed** (30.5–31.6 vs 20.5–21.6 ms). The A4 memory win (1.56x less KV at B = 8 / ctx 8192) is unchanged, but there is no speed win. Not worth it.

## Question

[[2026-11-18_1855_a4-g32-nibble-old-kv-plus-a8-g64-recent-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-u32-lut-vs-int8-weights-total-bytes-full-bitnetd-layer]] found that A4 g32 for tokens older than a 1024-token A8 window restores a 2.13x total-bytes edge, but attention ran at 0.78x of all-A8 because of the nibble unpack and f32 dequant. Its next question: can attention score Q·K directly on the nibble codes, with the per-32-dim scales folded into the scores (and the S·V side folded per group), so the 1.56x smaller KV also gives ≥ 1.5x faster multi-stream attention at B = 8 / ctx 8192?

## Method

- Attention-only microbench of the 18:55 KV layout (the weight path is unchanged and not re-timed): B ∈ {1, 8} streams × 5 KV heads × 4 q heads (GQA 20/5), HD 128, ctx ∈ {2048, 8192}, last 1024 tokens at A8 g64, older tokens at A4 g32 requantised from their A8 values. Online softmax per (stream, KV head) on a W = 8 thread pool, `OPENBLAS_NUM_THREADS=1`, tile sizes TK ∈ {1024, 2048} (best TK per variant used in comparisons).
- Old-region variants (the A8 window tile is identical in all):
  - `a8`: whole context A8 g64 dequant (control, 1.56x more KV bytes at ctx 8192).
  - `a4deq`: the 18:55 path: u16 LUT → int8 codes → f32 × scale → Q·Kᵀ, P·V (baseline to beat).
  - `a4fold`: byte → two f32 codes via a 256×2 f32 LUT (no int8 hop, no scale multiply); per-group partial scores from a block-diagonal Q (16 × 128) in one GEMM, then S = Σ_g S_g · k_scale[:, g]; V side multiplies P by v_scale per group and does four (G × T)·(T × 32) GEMMs.
  - `a4byte`: T-MAC-style scoring: a per-head table[pos, byte, q] = q · dec(byte) for the 64 byte positions, scores are 16 gathers + sum per group, scaled per group; V as in `a4fold`.
- Synthetic K/V: Gaussian with log-normal per-channel spread, 6 hot K channels × 6 and a slow positional drift (harsher than the 18:55 projected history). Error: per (stream, head) cos of the attention output vs exact f32-KV softmax; and max |Δ| vs `a4deq` to show the fused paths are the same math.
- 7 rounds × 3 reps per config, median (min–max); two independent runs.

## Command

```
cd /workspace/bench0255_1119
OPENBLAS_NUM_THREADS=1 python3 fused.py results_run1.json | tee run1.txt
OPENBLAS_NUM_THREADS=1 python3 fused.py results_run2.json > run2.txt
```

## Results (attention ms per decode step, all streams; run1 / run2 median)

| ctx / B | a8 (best TK) | a4deq (best TK) | a4fold (best TK) | a4byte (best TK) | KV MiB a8 / a4 | a4fold vs a4deq | a4fold vs a8 | a4byte vs a4deq |
|---|---|---|---|---|---|---|---|---|
| 2048 / 1 | 1.22 / 1.39 | 1.56 / 2.23 | 1.93 / 2.41 | 3.45 / 3.74 | 2.66 / 2.11 | 0.81x / 0.93x | 0.63x / 0.58x | 0.45x / 0.60x |
| 2048 / 8 | 7.25 / 8.59 | 11.82 / 11.70 | 12.63 / 11.98 | 17.77 / 16.69 | 21.25 / 16.88 | 0.94x / 0.98x | 0.57x / 0.72x | 0.67x / 0.70x |
| 8192 / 1 | 4.05 / 3.53 | 4.40 / 4.65 | 5.29 / 5.23 | 11.65 / 11.66 | 10.62 / 6.80 | 0.83x / 0.89x | 0.76x / 0.68x | 0.38x / 0.40x |
| 8192 / 8 | 21.59 / 20.50 | 30.48 / 31.61 | 39.01 / 39.90 | 60.47 / 64.00 | 85.00 / 54.38 | 0.78x / 0.79x | 0.55x / 0.51x | 0.50x / 0.49x |

(Ratio columns use each run's own best-TK medians.)

Spread at B = 8 / ctx 8192, TK 2048 (run1): a8 19.94–30.33, a4deq 29.14–31.55, a4fold 37.08–41.80, a4byte 59.65–65.57 ms.

Error (attention-output cos, min over streams × heads): a8 0.99477–0.99932; all three A4 paths identical to each other (max |Δ| vs a4deq ≤ 3.8e-6): 0.960 (2048/B1), 0.886 (2048/B8), 0.812 (8192). The A4 numbers are much lower than the 18:55 layer-delta cos (≥ 0.999997) because this run uses raw attention output on synthetic K with ×6 hot channels and no residual; that is a harsher metric, not a regression in the format, but it is a warning that A4 g32 K is sensitive to hot K channels.

## Verdict

**not-viable.** Folding the scales removes the T × 128 f32 multiply but adds a block-diagonal GEMM 4x wider in rows and four narrow V GEMMs, so it is 2–22% slower than plain dequant. Byte-LUT scoring replaces BLAS with 64 gathers per key and is 1.4–2.6x slower. In numpy the cost is the per-key unpack/gather itself, not the scale multiply; no A4 attention variant beats A8 (best 0.65–0.71x at B = 8 / ctx 8192). Keep A4 g32 old-token KV for its memory win (2.13x total bytes, 18:55) and accept 0.65–0.78x of A8 attention speed; a real speed win needs a compiled nibble dot kernel, which is outside this numpy lane.

## Links

- previous: [[2026-11-18_1855_a4-g32-nibble-old-kv-plus-a8-g64-recent-1024-window-multi-stream-decode-b1-b8-ctx-2048-8192-u32-lut-vs-int8-weights-total-bytes-full-bitnetd-layer]]
- related: [[2026-11-18_1455_multi-stream-batched-decode-b1-b4-b8-per-stream-a8-g64-kv-ctx-2048-8192-u32-lut-vs-int8-weights-total-bytes-full-bitnetd-layer]]
- this wave's pair files: none. Quant Chief 02:05 assignment was pair_A no gap and pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-19T09:12:00Z). Roadmap blobs: `JuniorHome docs/ROADMAP.md` 4efb666, `JuniorLLM docs/BETA_TO_OS.md` bac6923.
- bench dir: `/workspace/bench0255_1119` (fused.py, run1.txt, run2.txt, results_run1.json, results_run2.json)

## Script (`fused.py`)

```python
#!/usr/bin/env python3
"""Fused A4 g32 old-token attention without the f32 scale-dequant (02:55 11/19). Attention-only microbench of the
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
exact softmax. Usage: OPENBLAS_NUM_THREADS=1 python3 fused.py out.json"""
import json, sys, time, statistics
from concurrent.futures import ThreadPoolExecutor
import numpy as np
KVH, G, HD, WIN, W = 5, 4, 128, 1024, 8; SC = np.float32(1 / np.sqrt(HD)); NG = HD // 32
CTXS, BS, TKS = (2048, 8192), (1, 8), (1024, 2048)
ROUNDS, REPS = 7, 3
POOL = ThreadPoolExecutor(W)
rng = np.random.default_rng(1119)
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
    def bytes(s, fmt):
        n8 = s.n if fmt == "a8" else min(s.n, WIN); n4 = s.n - n8
        return 2 * KVH * (n8 * (HD + HD // 64 * 4) + n4 * (HD // 2 + NG * 4))
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
            for fmt in ("a8", "a4deq", "a4fold", "a4byte"):
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
            best = {f: min((res[(f, t)] for t in TKS), key=lambda r: r["med"]) for f in ("a8", "a4deq", "a4fold", "a4byte")}
            for f in ("a4fold", "a4byte"):
                for refn in ("a4deq", "a8"):
                    k = f"{f}_vs_{refn}_ctx{n}_B{B}"; r = best[refn]["med"] / best[f]["med"]
                    out["compare"][k] = dict(speed=r, bytes=best[refn]["mib"] / best[f]["mib"], new=best[f]["key"], ref=best[refn]["key"])
                    print(f"  {k:30s} speed {r:5.2f}x  bytes {best[refn]['mib'] / best[f]['mib']:5.2f}x  ({best[f]['key']} vs {best[refn]['key']})")
    json.dump(out, open(sys.argv[1], "w"), indent=1)
main()
```

## Next question

A4 g32 K lost the most accuracy on hot K channels here. Does keeping K at A8 and storing only V at A4 g32 for old tokens (K8V4) keep most of the bytes win (KV 1360 → 1080 B/token/layer) with attention speed close to all-A8, and is the B = 8 / ctx 8192 total-bytes edge over int8 weights still ≥ 1.5x?


close: sha256(prev || this)[:16] = 6aab9ec8ca5e9cdb
