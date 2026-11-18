---
local: true
date: 2026-11-17
wave: "2026-11-17 18:55 (ran ~18:45–18:52 MT; Chief last_run 2026-11-18T01:10:00Z ~18:10 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (prompt prefill)"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, prefill, attention, kv-cache, a8-g64, f16, gqa, u32-lut, int8, numpy, viable]
---

# Chunked prefill with real causal attention: at P = 4096 (C = 512) the **2-bit u32-LUT weights + A8 g64 KV** layer holds **21.9 MiB vs 76.3 MiB for int8 weights + f16 KV (3.48x fewer resident bytes)** at **0.98x of its total prefill time** (1397.9 vs 1371.7 ms run 1, 1389.7 vs 1367.3 ms run 2), with min per-token cos 0.999997 vs the exact f32-KV reference. Attention does **not** dominate before P = 2048: its share of the chunk step is 0.20–0.25 at ctx 2048 for every path and 0.31–0.37 at ctx 3584. Only the last chunk (ctx 4096) jumps: f16/A8 KV attention doubles (115–133 ms vs 52–68 at 3584; 0.48–0.55 share) while f32 KV stays at 61–64 ms (0.34–0.37 share).

## Question

[[2026-11-17_1455_prefill-crossover-chunk-c256-c512-c1024-c2048-p256-p1024-p2048-u32-lut-t256-t512-w4-w8-vs-int8-f32-full-bitnetd-layer]] settled prefill at C = 512 with 2-bit resident weights but ran ctx = 1 attention. Its next question, run here: at C = 512 chunks over a growing context (512 → 4096), how does KV-cache attention time compare to the 2-bit weight GEMMs per chunk, and does attention become the dominant term before P = 2048? And does the 2-bit memory win survive once the KV cache is counted?

Quant Chief `last_tips.json` `last_run` 2026-11-18T01:10:00Z (~18:10 MT): pair_A (Trit Pack & I2_S, Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock, Handshake & Ports) both **no gap**; roadmap blobs unchanged (4efb666 / bac6923); LLC JuniorHome 1c76370 left to LLC. So this note uses the lane's own question. JuniorLLM tip at push clone was 3b01409 (this lane, 14:51 MT); no Trit Pack, Handshake or LLC commit touched JuniorLLM in the last 6 h.

## Method

- **Unchanged:** `base.py`, `tiled.py`, `par.py`, `cache.py` byte-identical to `/workspace/bench1455_1117` (sha256 verified): hidden 2560, GQA 20 q / 5 KV heads × 128, FFN 6912, absmean ternary weights, seed 202611141055, fused q|k|v and gate|up, private per-worker scratch, `OPENBLAS_NUM_THREADS=1`, W = 8 workers.
- **Weights:** `u32` = 2-bit pack + u32-LUT take → f32 tile → BLAS (`par.fmm`, T = 256, the 14:55 pick); `int8` = int8-resident trit rows cast per tile (`cache.fmmc`, T = 128). Both exact W1.58.
- **KV cache:** `f32`; `f16` (cast to f32 for the whole context each chunk); `a8` = int8 per token per 64-dim group with f32 scale (dequant for the whole context each chunk). Bytes counted at 4096 tokens per layer.
- **Attention:** per KV head (its 4 q heads stacked), S = QKᵀ/√128 with causal mask (query c0+i sees keys ≤ c0+i), softmax, @V; the 5 heads run on the same 8-worker pool. No RoPE (cost-neutral for this question).
- **Prefill:** P = 4096 Gaussian tokens (seed 202611171855) in 8 chunks of C = 512; per chunk the weight-GEMM time (qkv + o + gate/up + down + norms/SiLU) and attention time (KV append/quant + dequant + scores + softmax + PV) are timed separately.
- **Timing:** 1 warm/reference pass per path, then 5 interleaved rounds (rotated order) × 2 full prefills = 10 per path; median (min–max) of totals, per-chunk medians. Two full runs. 8 cores, numpy only, offline, nothing bound, no downloads. Box had other load (load avg 4.2–5.6).
- **Error:** whole 4096-token output of each path vs `u32+f32` (exact reference; int8 weights are bit-identical trits).

Commands (in `/workspace/bench1855_1117`, `ctxattn.py` sha256 `01a62db6f8ef4f18…`):

```
OPENBLAS_NUM_THREADS=1 python3 ctxattn.py results_run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 python3 ctxattn.py results_run2.json > run2.txt
```

<details><summary>ctxattn.py</summary>

```python
#!/usr/bin/env python3
"""Chunked prefill with real causal GQA attention over a growing KV cache (18:55 11/17). P = 4096 prompt tokens in
C = 512 chunks through the full bitnetd layer (hidden 2560, 20 q / 5 KV heads x 128, FFN 6912). Weights: 2-bit pack +
u32-LUT expand (par.fmm, T=256 W=8, the 14:55 pick) or int8-resident trit rows (cache.fmmc, T=128 W=8); both exact W1.58.
KV cache: f32, f16 (cast to f32 per chunk) or A8 g64 (int8 per token per 64-dim group, f32 scale, dequant per chunk).
Attention = per KV head (4 q heads stacked) S = QK^T/sqrt(128) + causal mask, softmax, @V, 5 heads on the same W=8 pool.
Each chunk's weight-GEMM time and attention time (incl. KV append + dequant) are timed separately.
base/tiled/par/cache byte-identical to /workspace/bench1455_1117. Usage: OPENBLAS_NUM_THREADS=1 python3 ctxattn.py out.json"""
import json, sys, time, statistics
import numpy as np
import base, par, cache
from base import rms, silu, D, KVH, QH, HD
P_TOK, C, T8, TU, W = 4096, 512, 128, 256, 8
G = QH // KVH; KVD = KVH * HD; GS = 64
base.build(); u32_bytes = par.prep(); cache.prep()
int8_bytes = sum(cache.CT[k]["Q"].nbytes + cache.CT[k]["g"].nbytes for k in cache.CT)
for n in base.PW:
    for f in ("Wd", "Wt", "wm", "ws", "wm8", "ws8", "PT", "q8"): base.PW[n].pop(f, None)
MM = {"u32": lambda k, h: par.fmm(k, h, TU, W), "int8": lambda k, h: cache.fmmc(k, h, T8, W)}
class KV:
    def __init__(s, fmt):
        s.fmt = fmt; dt = {"f32": np.float32, "f16": np.float16, "a8": np.int8}[fmt]
        s.k = np.zeros((KVH, P_TOK, HD), dt); s.v = np.zeros((KVH, P_TOK, HD), dt); s.n = 0
        if fmt == "a8":
            s.ks = np.zeros((KVH, P_TOK, HD // GS), np.float32); s.vs = np.zeros((KVH, P_TOK, HD // GS), np.float32)
    def nbytes(s, n):
        b = 2 * KVH * n * HD * s.k.itemsize
        return b + (2 * KVH * n * (HD // GS) * 4 if s.fmt == "a8" else 0)
    def _q8(s, x):  # x (KVH, c, HD) f32 -> int8, scale
        g = x.reshape(KVH, -1, HD // GS, GS); m = np.abs(g).max(-1) / 127.0 + 1e-12
        return np.clip(np.rint(g / m[..., None]), -127, 127).astype(np.int8).reshape(x.shape), m.astype(np.float32)
    def append(s, k, v):  # k, v (c, KVD)
        c = k.shape[0]; a, b = s.n, s.n + c
        kh = k.reshape(c, KVH, HD).transpose(1, 0, 2); vh = v.reshape(c, KVH, HD).transpose(1, 0, 2)
        if s.fmt == "a8":
            s.k[:, a:b], s.ks[:, a:b] = s._q8(kh); s.v[:, a:b], s.vs[:, a:b] = s._q8(vh)
        else:
            s.k[:, a:b] = kh; s.v[:, a:b] = vh
        s.n = b
    def get(s, h):  # f32 (n, HD) K, V for head h
        n = s.n
        if s.fmt == "f32": return s.k[h, :n], s.v[h, :n]
        if s.fmt == "f16": return s.k[h, :n].astype(np.float32), s.v[h, :n].astype(np.float32)
        kk = (s.k[h, :n].reshape(n, -1, GS).astype(np.float32) * s.ks[h, :n, :, None]).reshape(n, HD)
        vv = (s.v[h, :n].reshape(n, -1, GS).astype(np.float32) * s.vs[h, :n, :, None]).reshape(n, HD)
        return kk, vv
def head(kv, qh, h, c0, out):
    K, V = kv.get(h); n = K.shape[0]; c = qh.shape[1]
    S = (qh.reshape(G * c, HD) @ K.T).reshape(G, c, n); S *= np.float32(1 / np.sqrt(HD))
    pos = c0 + np.arange(c)[:, None]; S[:, np.arange(n)[None, :] > pos] = -np.inf  # causal
    S -= S.max(-1, keepdims=True); np.exp(S, out=S); S /= S.sum(-1, keepdims=True)
    out[:, h] = (S.reshape(G * c, n) @ V).reshape(G, c, HD).transpose(1, 0, 2)
def attn(kv, q, k, v, c0):
    c = q.shape[0]; kv.append(k, v)
    qh = q.reshape(c, KVH, G, HD).transpose(1, 2, 0, 3)  # (KVH, G, c, HD)
    out = np.empty((c, KVH, G, HD), np.float32)
    for f in [par.pool(W).submit(head, kv, qh[h], h, c0, out) for h in range(KVH)]: f.result()
    return out.reshape(c, D)
def prefill(wk, kfmt, X, rec=None):
    mm = MM[wk]; kv = KV(kfmt); outs = []
    for c0 in range(0, X.shape[0], C):
        x = X[c0:c0 + C]; t0 = time.perf_counter()
        h1 = rms(x); q, k, v = mm("qkv", h1); t1 = time.perf_counter()
        a = attn(kv, q, k, v, c0); t2 = time.perf_counter()
        o = mm("o", a)[0]; x2 = x + o; h2 = rms(x2); g, u = mm("gu", h2); z = (silu(g) * u).astype(np.float32)
        y = x2 + mm("down", z)[0]; t3 = time.perf_counter()
        if rec is not None: rec.append(((t1 - t0) + (t3 - t2), t2 - t1))
        outs.append(y)
    return np.concatenate(outs), kv
PATHS = [("u32", "a8"), ("u32", "f16"), ("u32", "f32"), ("int8", "f16"), ("int8", "f32")]
def main():
    out = sys.argv[1]; rng = np.random.default_rng(202611171855)
    X = rng.standard_normal((P_TOK, D)).astype(np.float32)
    ref, _ = prefill("u32", "f32", X)
    res = {"u32_weight_bytes": u32_bytes, "int8_weight_bytes": int8_bytes, "paths": {}}
    for p in PATHS:
        y, kv = prefill(*p, X)
        d = (y - ref).astype(np.float64); r = ref.astype(np.float64)
        cs = np.sum(d * 0 + y * r, 1) / (np.linalg.norm(y, axis=1) * np.linalg.norm(r, axis=1))
        res["paths"]["%s+%s" % p] = dict(max_abs=float(np.abs(d).max()), rel_fro=float(np.linalg.norm(d) / np.linalg.norm(r)),
                                         min_cos=float(cs.min()), kv_bytes=kv.nbytes(P_TOK), runs=[])
    for rnd in range(5):
        order = PATHS[rnd % len(PATHS):] + PATHS[:rnd % len(PATHS)]
        for p in order:
            for _ in range(2):
                rec = []; t0 = time.perf_counter(); prefill(*p, X, rec); tt = time.perf_counter() - t0
                res["paths"]["%s+%s" % p]["runs"].append(dict(total=tt, chunks=rec))
    for name, R in res["paths"].items():
        tot = [r["total"] for r in R["runs"]]
        wt = [[r["chunks"][i][0] for r in R["runs"]] for i in range(P_TOK // C)]
        at = [[r["chunks"][i][1] for r in R["runs"]] for i in range(P_TOK // C)]
        R["total_med"] = statistics.median(tot); R["total_min"] = min(tot); R["total_max"] = max(tot)
        R["w_med"] = [statistics.median(c) for c in wt]; R["a_med"] = [statistics.median(c) for c in at]
        R.pop("runs")
        print(f"{name:10s} total {R['total_med']*1e3:7.1f} ms [{R['total_min']*1e3:.1f}-{R['total_max']*1e3:.1f}] "
              f"kv {R['kv_bytes']/2**20:.1f} MiB max_abs {R['max_abs']:.2e} rel {R['rel_fro']:.2e} mincos {R['min_cos']:.6f}")
        print("   chunk end ctx:", " ".join(f"{(i+1)*C}" for i in range(P_TOK // C)))
        print("   weights ms   :", " ".join(f"{w*1e3:.1f}" for w in R["w_med"]))
        print("   attn ms      :", " ".join(f"{a*1e3:.1f}" for a in R["a_med"]))
        print("   attn share   :", " ".join(f"{a/(a+w):.2f}" for w, a in zip(R["w_med"], R["a_med"])))
    print(f"weights u32 {u32_bytes/2**20:.1f} MiB int8 {int8_bytes/2**20:.1f} MiB")
    json.dump(res, open(out, "w"), indent=1)
main()
```

</details>

## Results

Raw output, run 1:

```
u32+a8     total  1397.9 ms [1178.3-1672.3] kv 5.3 MiB max_abs 9.48e-03 rel 1.48e-04 mincos 0.999997
   chunk end ctx: 512 1024 1536 2048 2560 3072 3584 4096
   weights ms   : 130.0 126.9 128.6 123.8 119.8 131.1 118.6 120.9
   attn ms      : 14.0 20.6 28.9 35.1 42.1 51.2 60.0 124.1
   attn share   : 0.10 0.14 0.18 0.22 0.26 0.28 0.34 0.51
u32+f16    total  1347.9 ms [1217.7-1759.5] kv 10.0 MiB max_abs 3.44e-04 rel 5.26e-06 mincos 1.000000
   chunk end ctx: 512 1024 1536 2048 2560 3072 3584 4096
   weights ms   : 115.0 121.9 111.0 113.7 126.3 125.9 118.3 126.8
   attn ms      : 12.3 18.3 25.1 33.1 43.0 49.4 58.0 118.8
   attn share   : 0.10 0.13 0.18 0.23 0.25 0.28 0.33 0.48
u32+f32    total  1285.8 ms [1151.2-1362.9] kv 20.0 MiB max_abs 0.00e+00 rel 0.00e+00 mincos 1.000000
   chunk end ctx: 512 1024 1536 2048 2560 3072 3584 4096
   weights ms   : 124.0 121.5 119.2 118.1 116.3 115.2 114.6 119.0
   attn ms      : 10.7 16.3 24.1 33.5 38.6 43.8 54.2 61.1
   attn share   : 0.08 0.12 0.17 0.22 0.25 0.28 0.32 0.34
int8+f16   total  1371.7 ms [1160.8-1534.5] kv 10.0 MiB max_abs 3.44e-04 rel 5.26e-06 mincos 1.000000
   chunk end ctx: 512 1024 1536 2048 2560 3072 3584 4096
   weights ms   : 123.4 120.3 116.4 108.7 112.7 109.5 112.5 107.9
   attn ms      : 13.1 21.5 28.8 34.6 40.0 50.1 60.3 132.6
   attn share   : 0.10 0.15 0.20 0.24 0.26 0.31 0.35 0.55
int8+f32   total  1231.9 ms [1129.2-1456.0] kv 20.0 MiB max_abs 0.00e+00 rel 0.00e+00 mincos 1.000000
   chunk end ctx: 512 1024 1536 2048 2560 3072 3584 4096
   weights ms   : 123.6 121.8 116.1 107.9 111.3 107.7 115.7 118.7
   attn ms      : 10.7 19.3 23.7 31.7 42.0 45.5 52.4 62.9
   attn share   : 0.08 0.14 0.17 0.23 0.27 0.30 0.31 0.35
weights u32 16.6 MiB int8 66.3 MiB
```

Raw output, run 2:

```
u32+a8     total  1389.7 ms [1295.2-1565.4] kv 5.3 MiB max_abs 9.48e-03 rel 1.48e-04 mincos 0.999997
   chunk end ctx: 512 1024 1536 2048 2560 3072 3584 4096
   weights ms   : 123.6 115.4 125.3 115.0 128.7 124.8 117.2 118.5
   attn ms      : 13.1 20.5 29.0 35.0 42.4 54.0 68.3 115.1
   attn share   : 0.10 0.15 0.19 0.23 0.25 0.30 0.37 0.49
u32+f16    total  1388.8 ms [1181.3-1640.4] kv 10.0 MiB max_abs 3.44e-04 rel 5.26e-06 mincos 1.000000
   chunk end ctx: 512 1024 1536 2048 2560 3072 3584 4096
   weights ms   : 117.8 120.5 116.3 129.6 122.9 118.5 126.5 121.4
   attn ms      : 12.1 18.7 26.0 32.2 42.9 52.2 56.8 122.9
   attn share   : 0.09 0.13 0.18 0.20 0.26 0.31 0.31 0.50
u32+f32    total  1354.8 ms [1184.9-1519.7] kv 20.0 MiB max_abs 0.00e+00 rel 0.00e+00 mincos 1.000000
   chunk end ctx: 512 1024 1536 2048 2560 3072 3584 4096
   weights ms   : 131.8 125.3 125.0 121.4 124.9 124.2 121.6 126.2
   attn ms      : 10.2 17.1 25.2 33.4 37.4 48.8 56.0 64.4
   attn share   : 0.07 0.12 0.17 0.22 0.23 0.28 0.32 0.34
int8+f16   total  1367.3 ms [1179.8-1515.0] kv 10.0 MiB max_abs 3.44e-04 rel 5.26e-06 mincos 1.000000
   chunk end ctx: 512 1024 1536 2048 2560 3072 3584 4096
   weights ms   : 114.8 122.7 112.3 106.9 121.2 117.7 112.2 105.0
   attn ms      : 12.8 19.9 28.5 36.3 41.2 53.7 53.8 116.2
   attn share   : 0.10 0.14 0.20 0.25 0.25 0.31 0.32 0.53
int8+f32   total  1191.8 ms [1051.3-1368.7] kv 20.0 MiB max_abs 0.00e+00 rel 0.00e+00 mincos 1.000000
   chunk end ctx: 512 1024 1536 2048 2560 3072 3584 4096
   weights ms   : 103.0 112.5 110.7 108.9 111.6 101.7 114.7 102.6
   attn ms      : 10.1 17.6 23.4 31.2 41.4 44.2 51.7 60.8
   attn share   : 0.09 0.14 0.17 0.22 0.27 0.30 0.31 0.37
weights u32 16.6 MiB int8 66.3 MiB
```

**Summary (P = 4096, C = 512; bytes per layer = weights + KV at 4096 tokens):**

| path | resident MiB (w + kv) | total ms run1 [min–max] | total ms run2 [min–max] | vs int8+f16 speed (r1 / r2) | bytes vs int8+f16 | max abs / min cos |
|---|---|---|---|---|---|---|
| `u32+a8` | 16.6 + 5.3 = **21.9** | 1397.9 [1178–1672] | 1389.7 [1295–1565] | 0.98 / 0.98 | **3.48x fewer** | 9.5e-3 / 0.999997 |
| `u32+f16` | 16.6 + 10.0 = 26.6 | 1347.9 [1218–1760] | 1388.8 [1181–1640] | 1.02 / 0.98 | 2.87x fewer | 3.4e-4 / 1.000000 |
| `u32+f32` | 16.6 + 20.0 = 36.6 | 1285.8 [1151–1363] | 1354.8 [1185–1520] | 1.07 / 1.01 | 2.08x fewer | exact |
| `int8+f16` | 66.3 + 10.0 = 76.3 | 1371.7 [1161–1535] | 1367.3 [1180–1515] | 1.00 / 1.00 | 1.00 | 3.4e-4 / 1.000000 |
| `int8+f32` | 66.3 + 20.0 = 86.3 | 1231.9 [1129–1456] | 1191.8 [1051–1369] | 1.11 / 1.15 | 0.88x | exact |

**Attention share of the chunk step (median over paths and runs):** ctx 512 0.07–0.10, 1024 0.12–0.15, 2048 0.20–0.25, 3072 0.28–0.31, 3584 0.31–0.37, 4096 0.34–0.37 (f32 KV) / 0.48–0.55 (f16, A8 KV).

## Verdict

**viable** (memory, ≥ 1.5x at parity speed). Counting the KV cache at 4096 tokens, 2-bit weights + A8 g64 KV need 3.48x fewer bytes per layer than int8 weights + f16 KV and run the whole 4096-token prefill at 0.98x of its speed (both runs), at min cos 0.999997. Against the fastest control, int8 weights + f32 KV, it is 3.94x fewer bytes at 0.85–0.88x speed. Attention is not the dominant term before P = 2048 (≤ 0.25 of the step), so prefill cost stays weight-bound through 2048 and 2-bit resident weights remain the right default for i2sd. The one cliff is the per-chunk whole-context cast/dequant of f16 or A8 KV at ctx 4096, which doubles attention time there (and is why f32 KV wins on speed at 4096); that is a KV-kernel issue, not a weight-format one.

## Links

- previous: [[2026-11-17_1455_prefill-crossover-chunk-c256-c512-c1024-c2048-p256-p1024-p2048-u32-lut-t256-t512-w4-w8-vs-int8-f32-full-bitnetd-layer]]
- this wave's pair files: none — Quant Chief 18:05 assignment pair_A no gap, pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-18T01:10:00Z); roadmap `JuniorHome docs/ROADMAP.md` blob 4efb666, `JuniorLLM docs/BETA_TO_OS.md` blob bac6923
- bench dir: `/workspace/bench1855_1117` (run1.txt, run2.txt, results_run1.json, results_run2.json)

## Next question

Remove the ctx-4096 KV cliff: does a context-tiled (flash-style, 1024-key tiles with running max/sum) attention that dequantizes A8 g64 K/V one tile at a time keep the attention share at or below the f32-KV level (≈ 0.35 at 4096) up to ctx 8192, while keeping A8's 2x KV-byte cut over f16?

close: sha256(prev || this)[:16] = 5942d4fe0c821bf7
