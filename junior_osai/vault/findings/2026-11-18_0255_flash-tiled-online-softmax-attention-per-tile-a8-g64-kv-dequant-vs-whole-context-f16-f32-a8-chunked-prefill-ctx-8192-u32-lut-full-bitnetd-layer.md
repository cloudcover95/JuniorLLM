---
local: true
date: 2026-11-18
wave: "2026-11-18 02:55 (ran ~02:50–02:57 MT; Chief last_run 2026-11-18T09:10:00Z ~02:10 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (long-prompt prefill)"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, prefill, attention, flash-attention, online-softmax, kv-cache, a8-g64, f16, gqa, u32-lut, ctx-8192, numpy, viable]
---

# Context-tiled (flash-style) attention removes the ctx > 4096 KV cliff: in an 8192-token chunked prefill (C = 512) with 2-bit u32-LUT weights, **A8 g64 KV dequantized one 1024-key tile at a time** spends **776 / 802 ms in attention vs 1913 / 1626 ms for the 18:55 whole-context f16 kernel (2.47x / 2.03x faster, runs 1 / 2)** and 1893 / 1527 ms for whole-context A8 (2.44x / 1.90x), with **10.6 MiB of KV vs 20.0 MiB f16 (1.89x fewer)** and the same error as whole-context A8 (min cos 0.999996). Whole prefill: 2717 / 2533 ms vs 3947 / 3338 ms (1.45x / 1.32x). The last chunk (ctx 8192) drops from 306 / 214 ms attention to 80 / 82 ms. Attention share at ctx 8192 falls from 0.68–0.73 to 0.42–0.45 and now grows smoothly with context; it stays ≤ 0.35 only through ctx ≈ 5120, so the 0.35-up-to-8192 target is not fully met.

## Question

[[2026-11-17_1855_chunked-prefill-causal-gqa-attention-ctx-512-4096-a8-g64-f16-f32-kv-share-vs-u32-lut-and-int8-weights-full-bitnetd-layer]] found that whole-context dequant/cast of f16 or A8 KV doubles attention time at ctx 4096. Its next question, run here: does a context-tiled (flash-style, 1024-key tiles with running max/sum) attention that dequantizes A8 g64 K/V one tile at a time keep the attention share at or below the f32-KV level (≈ 0.35 at 4096) up to ctx 8192, while keeping A8's 2x KV-byte cut over f16?

Quant Chief `last_tips.json` `last_run` 2026-11-18T09:10:00Z (~02:10 MT): pair_A (Trit Pack & I2_S, Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock, Handshake & Ports) both **no gap**; roadmap blobs unchanged (4efb666 / bac6923); LLC JuniorHome 1e18f49 left to LLC. So this note uses the lane's own question. JuniorLLM tip at push clone was fd72daf (02:49 MT, cloudcover95, "rails: live=false guard", adds only `rails/livefalse`); no Trit Pack, Handshake or LLC commit touched `junior_osai/vault/findings/` in the last 6 h.

## Method

- **Unchanged:** `base.py`, `tiled.py`, `par.py`, `cache.py` byte-identical to `/workspace/bench1855_1117` (sha256 verified): hidden 2560, GQA 20 q / 5 KV heads × 128, FFN 6912, absmean ternary weights, fused q|k|v and gate|up, `OPENBLAS_NUM_THREADS=1`, W = 8 workers. Weights = 2-bit pack + u32-LUT expand (`par.fmm`, T = 256), exact W1.58, 16.6 MiB/layer.
- **KV cache:** `f32`, `f16`, `a8` = int8 per token per 64-dim group + f32 scale (same quantizer as 18:55). Bytes at 8192 tokens per layer.
- **Kernels** (per KV head, its 4 q heads stacked, 5 heads on the W = 8 pool):
  - `whole` — the 18:55 kernel: dequant/cast the whole context, one S = QKᵀ/√128 over all n keys, boolean causal mask, softmax, @V.
  - `flashT` — key tiles of T ∈ {512, 1024, 2048}; only tiles up to the chunk's last query; dequant/cast **one tile**, S tile, causal mask only on tiles crossing the chunk diagonal, online softmax (running max m, sum l, rescale acc), divide by l at the end.
- **Prefill:** P = 8192 Gaussian tokens (seed 202611180255) in 16 chunks of C = 512; per chunk the weight time (qkv + o + gate/up + down + norms/SiLU) and attention time (KV append/quant + dequant + scores + softmax + PV) are timed separately.
- **Timing:** 1 warm/error pass per path, then 4 rounds (rotated order) × 1 full prefill = 4 per path; median (min–max) of totals and of summed attention, per-chunk medians. Two full runs (141 s and 126 s wall). 8 cores, numpy only, offline, nothing bound, no downloads. Load avg 1.0 at start, ~4.8 during (the bench itself).
- **Error:** whole 8192-token output vs `f32+whole` (exact reference).

Commands (in `/workspace/bench0255_1118`, `flashattn.py` sha256 `a9a9f003d494226c…`):

```
OPENBLAS_NUM_THREADS=1 python3 flashattn.py results_run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 python3 flashattn.py results_run2.json > run2.txt
```

<details><summary>flashattn.py</summary>

```python
#!/usr/bin/env python3
"""Context-tiled (flash-style) attention with per-tile KV dequant vs whole-context dequant (02:55 11/18).
Chunked prefill P = 8192 tokens in C = 512 chunks through the full bitnetd layer (hidden 2560, 20 q / 5 KV heads x 128,
FFN 6912), weights = 2-bit pack + u32-LUT expand (par.fmm, T=256, W=8; exact W1.58, the 14:55/18:55 pick).
KV: f32 / f16 / A8 g64 (int8 per token per 64-dim group, f32 scale). Attention per KV head (4 q heads stacked), causal.
  whole : 18:55 kernel - dequant/cast the whole context, one S = QK^T over all keys, boolean causal mask, softmax, @V.
  flashT: key tiles of T tokens; only tiles <= last query pos; dequant/cast ONE tile, S tile, causal mask only on tiles
          that cross the chunk diagonal, running max m / sum l / acc rescale (online softmax), divide at the end.
Weight GEMM time and attention time per chunk are timed separately. base/par/tiled/cache byte-identical to
/workspace/bench1855_1117. Usage: OPENBLAS_NUM_THREADS=1 python3 flashattn.py out.json"""
import json, sys, time, statistics
import numpy as np
import base, par
from base import rms, silu, D, KVH, QH, HD
P_TOK, C, TU, W = 8192, 512, 256, 8
G = QH // KVH; GS = 64; SC = np.float32(1 / np.sqrt(HD))
base.build(); u32_bytes = par.prep()
for n in base.PW:
    for f in ("Wd", "Wt", "wm", "ws", "wm8", "ws8", "PT", "q8"): base.PW[n].pop(f, None)
mm = lambda k, h: par.fmm(k, h, TU, W)
class KV:
    def __init__(s, fmt):
        s.fmt = fmt; dt = {"f32": np.float32, "f16": np.float16, "a8": np.int8}[fmt]
        s.k = np.zeros((KVH, P_TOK, HD), dt); s.v = np.zeros((KVH, P_TOK, HD), dt); s.n = 0
        if fmt == "a8":
            s.ks = np.zeros((KVH, P_TOK, HD // GS), np.float32); s.vs = np.zeros((KVH, P_TOK, HD // GS), np.float32)
    def nbytes(s, n):
        b = 2 * KVH * n * HD * s.k.itemsize
        return b + (2 * KVH * n * (HD // GS) * 4 if s.fmt == "a8" else 0)
    def _q8(s, x):
        g = x.reshape(KVH, -1, HD // GS, GS); m = np.abs(g).max(-1) / 127.0 + 1e-12
        return np.clip(np.rint(g / m[..., None]), -127, 127).astype(np.int8).reshape(x.shape), m.astype(np.float32)
    def append(s, k, v):
        c = k.shape[0]; a, b = s.n, s.n + c
        kh = k.reshape(c, KVH, HD).transpose(1, 0, 2); vh = v.reshape(c, KVH, HD).transpose(1, 0, 2)
        if s.fmt == "a8":
            s.k[:, a:b], s.ks[:, a:b] = s._q8(kh); s.v[:, a:b], s.vs[:, a:b] = s._q8(vh)
        else:
            s.k[:, a:b] = kh; s.v[:, a:b] = vh
        s.n = b
    def get(s, h, a, b):  # f32 (b-a, HD) K, V for head h, tokens [a, b)
        if s.fmt == "f32": return s.k[h, a:b], s.v[h, a:b]
        if s.fmt == "f16": return s.k[h, a:b].astype(np.float32), s.v[h, a:b].astype(np.float32)
        n = b - a
        kk = (s.k[h, a:b].reshape(n, -1, GS).astype(np.float32) * s.ks[h, a:b, :, None]).reshape(n, HD)
        vv = (s.v[h, a:b].reshape(n, -1, GS).astype(np.float32) * s.vs[h, a:b, :, None]).reshape(n, HD)
        return kk, vv
def head_whole(kv, qh, h, c0, out):
    K, V = kv.get(h, 0, kv.n); n = K.shape[0]; c = qh.shape[1]
    S = (qh.reshape(G * c, HD) @ K.T).reshape(G, c, n); S *= SC
    pos = c0 + np.arange(c)[:, None]; S[:, np.arange(n)[None, :] > pos] = -np.inf
    S -= S.max(-1, keepdims=True); np.exp(S, out=S); S /= S.sum(-1, keepdims=True)
    out[:, h] = (S.reshape(G * c, n) @ V).reshape(G, c, HD).transpose(1, 0, 2)
def make_flash(T):
    def head_flash(kv, qh, h, c0, out):
        c = qh.shape[1]; Q = qh.reshape(G * c, HD) * SC; n = kv.n
        m = np.full((G * c, 1), -np.inf, np.float32); l = np.zeros((G * c, 1), np.float32)
        acc = np.zeros((G * c, HD), np.float32)
        qpos = np.tile(c0 + np.arange(c), G)[:, None]
        for a in range(0, n, T):
            b = min(a + T, n); K, V = kv.get(h, a, b)
            S = Q @ K.T
            if b - 1 > c0:  # tile crosses the diagonal of this chunk -> mask future keys
                S[np.arange(a, b)[None, :] > qpos] = -np.inf
            mn = np.maximum(m, S.max(1, keepdims=True)); al = np.exp(m - mn)
            S -= mn; np.exp(S, out=S)
            l *= al; l += S.sum(1, keepdims=True); acc *= al; acc += S @ V; m = mn
        acc /= l
        out[:, h] = acc.reshape(G, c, HD).transpose(1, 0, 2)
    return head_flash
KERN = {"whole": head_whole, "flash1024": make_flash(1024), "flash512": make_flash(512), "flash2048": make_flash(2048)}
def attn(kv, kern, q, k, v, c0):
    c = q.shape[0]; kv.append(k, v)
    qh = q.reshape(c, KVH, G, HD).transpose(1, 2, 0, 3)
    out = np.empty((c, KVH, G, HD), np.float32)
    for f in [par.pool(W).submit(KERN[kern], kv, qh[h], h, c0, out) for h in range(KVH)]: f.result()
    return out.reshape(c, D)
def prefill(kfmt, kern, X, rec=None):
    kv = KV(kfmt); outs = []
    for c0 in range(0, X.shape[0], C):
        x = X[c0:c0 + C]; t0 = time.perf_counter()
        h1 = rms(x); q, k, v = mm("qkv", h1); t1 = time.perf_counter()
        a = attn(kv, kern, q, k, v, c0); t2 = time.perf_counter()
        o = mm("o", a)[0]; x2 = x + o; h2 = rms(x2); g, u = mm("gu", h2); z = (silu(g) * u).astype(np.float32)
        y = x2 + mm("down", z)[0]; t3 = time.perf_counter()
        if rec is not None: rec.append(((t1 - t0) + (t3 - t2), t2 - t1))
        outs.append(y)
    return np.concatenate(outs), kv
PATHS = [("f32", "whole"), ("f16", "whole"), ("a8", "whole"), ("f32", "flash1024"), ("f16", "flash1024"),
         ("a8", "flash512"), ("a8", "flash1024"), ("a8", "flash2048")]
ROUNDS = 4
def main():
    out = sys.argv[1]; rng = np.random.default_rng(202611180255)
    X = rng.standard_normal((P_TOK, D)).astype(np.float32)
    ref, _ = prefill("f32", "whole", X)
    res = {"u32_weight_bytes": u32_bytes, "P": P_TOK, "C": C, "rounds": ROUNDS, "paths": {}}
    for p in PATHS:
        y, kv = prefill(*p, X)
        d = (y - ref).astype(np.float64); r = ref.astype(np.float64)
        cs = np.sum(y * r, 1) / (np.linalg.norm(y, axis=1) * np.linalg.norm(r, axis=1))
        res["paths"]["%s+%s" % p] = dict(max_abs=float(np.abs(d).max()), rel_fro=float(np.linalg.norm(d) / np.linalg.norm(r)),
                                         min_cos=float(cs.min()), kv_bytes=kv.nbytes(P_TOK), runs=[])
    for rnd in range(ROUNDS):
        k0 = (rnd * 3) % len(PATHS); order = PATHS[k0:] + PATHS[:k0]
        for p in order:
            rec = []; t0 = time.perf_counter(); prefill(*p, X, rec); tt = time.perf_counter() - t0
            res["paths"]["%s+%s" % p]["runs"].append(dict(total=tt, chunks=rec))
    NC = P_TOK // C
    for name, R in res["paths"].items():
        tot = [r["total"] for r in R["runs"]]
        wt = [[r["chunks"][i][0] for r in R["runs"]] for i in range(NC)]
        at = [[r["chunks"][i][1] for r in R["runs"]] for i in range(NC)]
        atot = [sum(ch[1] for ch in r["chunks"]) for r in R["runs"]]
        R["total_med"] = statistics.median(tot); R["total_min"] = min(tot); R["total_max"] = max(tot)
        R["attn_total_med"] = statistics.median(atot); R["attn_total_min"] = min(atot); R["attn_total_max"] = max(atot)
        R["w_med"] = [statistics.median(c) for c in wt]; R["a_med"] = [statistics.median(c) for c in at]
        R.pop("runs")
        print(f"{name:14s} total {R['total_med']*1e3:7.1f} ms [{R['total_min']*1e3:.1f}-{R['total_max']*1e3:.1f}] "
              f"attn {R['attn_total_med']*1e3:7.1f} ms [{R['attn_total_min']*1e3:.1f}-{R['attn_total_max']*1e3:.1f}] "
              f"kv {R['kv_bytes']/2**20:.1f} MiB max_abs {R['max_abs']:.2e} rel {R['rel_fro']:.2e} mincos {R['min_cos']:.6f}")
        print("   ctx  :", " ".join(f"{(i+1)*C:5d}" for i in range(NC)))
        print("   w ms :", " ".join(f"{w*1e3:5.0f}" for w in R["w_med"]))
        print("   a ms :", " ".join(f"{a*1e3:5.0f}" for a in R["a_med"]))
        print("   share:", " ".join(f"{a/(a+w):5.2f}" for w, a in zip(R["w_med"], R["a_med"])))
    print(f"weights u32 {u32_bytes/2**20:.1f} MiB")
    json.dump(res, open(out, "w"), indent=1)
main()
```

</details>

## Results

| path | KV MiB @8192 | attn total ms r1 [min–max] | attn total ms r2 [min–max] | prefill total ms r1 [min–max] | prefill total ms r2 [min–max] | attn ms @ctx 8192 (r1 / r2) | share @4096 / @8192 (r1; r2) | max abs / min cos |
|---|---|---|---|---|---|---|---|---|
| `f32+whole` (ref) | 40.0 | 1908.7 [1727–2025] | 1499.0 [1387–1771] | 3919.2 [3547–4000] | 3196.3 [3005–3688] | 310 / 229 | 0.34/0.73; 0.37/0.69 | exact |
| `f16+whole` (18:55 control) | 20.0 | 1913.4 [1791–2130] | 1625.9 [1471–1820] | 3947.2 [3597–4172] | 3337.6 [3138–3888] | 292 / 234 | 0.37/0.71; 0.37/0.68 | 3.6e-4 / 1.000000 |
| `a8+whole` | 10.6 | 1892.5 [1706–2010] | 1527.1 [1453–1696] | 3895.8 [3432–4113] | 3161.7 [3092–3628] | 306 / 214 | 0.35/0.72; 0.38/0.68 | 1.2e-2 / 0.999996 |
| `f32+flash1024` | 40.0 | 779.2 [680–825] | 697.7 [671–805] | 2777.9 [2385–2889] | 2408.8 [2272–2772] | 85 / 71 | 0.28/0.45; 0.30/0.42 | 7.8e-7 / 1.000000 |
| `f16+flash1024` | 20.0 | 809.8 [757–878] | 739.8 [705–776] | 2822.6 [2566–2949] | 2416.7 [2353–2660] | 85 / 78 | 0.31/0.44; 0.30/0.45 | 3.6e-4 / 1.000000 |
| `a8+flash512` | 10.6 | 779.6 [761–824] | 759.1 [730–775] | 2696.1 [2514–2803] | 2415.4 [2320–2531] | 89 / 83 | 0.29/0.43; 0.30/0.45 | 1.2e-2 / 0.999996 |
| **`a8+flash1024`** | **10.6** | **776.1 [745–802]** | **801.7 [732–834]** | **2717.2 [2449–2816]** | **2532.5 [2372–2714]** | **80 / 82** | 0.30/0.42; 0.32/0.45 | 1.2e-2 / 0.999996 |
| `a8+flash2048` | 10.6 | 891.1 [858–961] | 849.9 [780–868] | 2853.6 [2660–2948] | 2613.2 [2402–2672] | 97 / 86 | 0.33/0.47; 0.33/0.47 | 1.2e-2 / 0.999996 |

Weight time per chunk is flat at ~100–127 ms for every path (u32 weights 16.6 MiB). Per-chunk attention ms (run 2): whole kernels jump at ctx 4608–5120 (57 → 66–115 → 91–141 ms) and reach 214–234 ms at 8192; flash kernels grow linearly (≈ 9–11 ms per 512 keys) to 71–86 ms at 8192.

**Ratios (`a8+flash1024`, run 1 / run 2):** attention vs `f16+whole` **2.47x / 2.03x**, vs `a8+whole` 2.44x / 1.90x; whole prefill vs `f16+whole` 1.45x / 1.32x; KV bytes vs f16 **1.89x fewer**, vs f32 3.77x fewer; attention vs `f32+flash1024` 1.00x / 0.87x (A8 dequant per tile costs ≤ 13%). Per-layer resident bytes at 8192 (weights + KV): 2-bit + A8 27.2 MiB vs 2-bit + f16 36.6 MiB vs int8 weights + f16 KV 86.3 MiB (3.17x). Transient score/dequant buffer per head (analytic): whole = 4·512·8192·4 B = 64 MiB scores + 8 MiB f32 K/V; flash1024 = 8 MiB scores + 1 MiB tile (≈ 8x smaller, ×5 concurrent heads: ~360 MiB → ~45 MiB), which is the likely cause of the whole-kernel cliff once S exceeds cache. Tile 512 ≈ 1024 within noise; 2048 is ~6–15% slower.

<details><summary>run2.txt (run 1 in results_run1.json / run1.txt)</summary>

```
f32+whole      total  3196.3 ms [3005.0-3688.2] attn  1499.0 ms [1386.5-1770.9] kv 40.0 MiB max_abs 0.00e+00 rel 0.00e+00 mincos 1.000000
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :    99   102    98   100   105   106   100    95    97    95   105    95   100   101   109   103
   a ms :     9    15    20    27    33    39    45    57    66    91   141   153   160   206   207   229
   share:  0.08  0.13  0.17  0.22  0.24  0.27  0.31  0.37  0.41  0.49  0.57  0.62  0.62  0.67  0.66  0.69
f16+whole      total  3337.6 ms [3138.0-3888.3] attn  1625.9 ms [1471.1-1820.4] kv 20.0 MiB max_abs 3.62e-04 rel 3.93e-06 mincos 1.000000
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :   102   101   101   121   109   105   101    96   101   105   103   100   100   108   105   109
   a ms :    12    17    24    36    38    48    49    57    87   109   129   162   178   209   203   234
   share:  0.11  0.14  0.19  0.23  0.26  0.31  0.33  0.37  0.46  0.51  0.56  0.62  0.64  0.66  0.66  0.68
a8+whole       total  3161.7 ms [3091.7-3627.6] attn  1527.1 ms [1453.4-1695.6] kv 10.6 MiB max_abs 1.20e-02 rel 1.13e-04 mincos 0.999996
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :   103   102   103   100    98    96   103    95    96   100    97   107   100    98   102   100
   a ms :    12    18    23    30    36    43    52    58    73   115   123   149   186   186   219   214
   share:  0.10  0.15  0.19  0.23  0.27  0.31  0.33  0.38  0.43  0.53  0.56  0.58  0.65  0.66  0.68  0.68
f32+flash1024  total  2408.8 ms [2271.7-2771.5] attn   697.7 ms [670.8-805.4] kv 40.0 MiB max_abs 7.75e-07 rel 9.78e-08 mincos 1.000000
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :   100   103   110   104   104   106   102    97    99    96    95   100   104    99   109   100
   a ms :     9    14    21    25    29    35    35    43    46    48    51    57    61    67    73    71
   share:  0.08  0.12  0.16  0.19  0.22  0.25  0.26  0.30  0.32  0.34  0.35  0.36  0.37  0.40  0.40  0.42
f16+flash1024  total  2416.7 ms [2352.7-2659.7] attn   739.8 ms [705.1-776.1] kv 20.0 MiB max_abs 3.62e-04 rel 3.93e-06 mincos 1.000000
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :    97   109    97    96   101    93    94    99   100   100   105   105   109   116    99    98
   a ms :    10    16    20    25    30    34    38    42    47    55    58    63    65    71    75    78
   share:  0.09  0.13  0.17  0.21  0.23  0.27  0.29  0.30  0.32  0.36  0.36  0.37  0.38  0.38  0.43  0.45
a8+flash512    total  2415.4 ms [2320.1-2531.3] attn   759.1 ms [729.5-774.5] kv 10.6 MiB max_abs 1.20e-02 rel 1.13e-04 mincos 0.999996
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :    92    93    94    91    98    97    99    97    93   103   101   107   102    99   102   100
   a ms :    11    16    20    27    29    35    39    42    47    56    58    64    67    75    77    83
   share:  0.11  0.15  0.17  0.23  0.23  0.26  0.28  0.30  0.34  0.35  0.36  0.38  0.39  0.43  0.43  0.45
a8+flash1024   total  2532.5 ms [2372.3-2714.3] attn   801.7 ms [732.0-834.1] kv 10.6 MiB max_abs 1.20e-02 rel 1.13e-04 mincos 0.999996
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :   101   111   102    95    98    98   101   104   107   100   107   104    96   104   102   102
   a ms :    11    16    23    26    32    36    42    49    49    57    62    66    65    72    83    82
   share:  0.10  0.13  0.18  0.21  0.24  0.27  0.30  0.32  0.32  0.37  0.37  0.39  0.40  0.41  0.45  0.45
a8+flash2048   total  2613.2 ms [2401.6-2672.2] attn   849.9 ms [779.7-867.6] kv 10.6 MiB max_abs 1.20e-02 rel 1.13e-04 mincos 0.999996
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :   110   108   103   110   110   106   103   101   113   109   101   100    99   100    99    96
   a ms :    11    16    24    29    34    38    47    51    57    61    68    67    73    78    86    86
   share:  0.09  0.13  0.19  0.21  0.24  0.26  0.31  0.33  0.34  0.36  0.40  0.40  0.43  0.44  0.46  0.47
weights u32 16.6 MiB
```

</details>

## Verdict

**viable** (speed, ≥ 1.5x on attention in both runs at unchanged error, keeping A8's 1.89x KV-byte cut). Tiling the context in 1024-key tiles with online softmax and dequantizing A8 g64 K/V per tile makes 8192-token prefill attention 2.03–2.47x faster than the 18:55 whole-context f16 kernel and 1.90–2.44x faster than whole-context A8, with error identical to whole-context A8 (max abs 1.2e-2, min cos 0.999996; the tiling itself adds ≤ 7.8e-7 vs exact f32). A8 + flash matches f32 + flash speed within 0–13% while holding 3.77x fewer KV bytes, so A8 g64 stays the KV default and the 4096 cliff was a kernel problem, not a format one. End-to-end prefill is 1.32–1.45x faster (just under the bar on its own). The stated target is only partly met: attention share is ≤ 0.35 through ctx ≈ 5120 but reaches 0.42–0.45 at 8192, the same as f32 + flash, because attention per chunk is now honestly linear in context.

## Links

- previous: [[2026-11-17_1855_chunked-prefill-causal-gqa-attention-ctx-512-4096-a8-g64-f16-f32-kv-share-vs-u32-lut-and-int8-weights-full-bitnetd-layer]]
- this wave's pair files: none — Quant Chief 02:05 assignment pair_A no gap, pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-18T09:10:00Z); roadmap `JuniorHome docs/ROADMAP.md` blob 4efb666, `JuniorLLM docs/BETA_TO_OS.md` blob bac6923
- bench dir: `/workspace/bench0255_1118` (flashattn.py, run1.txt, run2.txt, results_run1.json, results_run2.json)

## Next question

Use the idle workers at long context: with only 5 KV heads on 8 workers, does splitting each head's key tiles across workers (flash-decoding style split-KV, per-split m/l/acc merged by log-sum-exp) bring the A8 + flash1024 attention share at ctx 8192 from ~0.43 down to ≤ 0.35 (≥ 1.3x attention speed-up) without changing the error?

close: sha256(prev || this)[:16] = b55f72b2c7eecff8
