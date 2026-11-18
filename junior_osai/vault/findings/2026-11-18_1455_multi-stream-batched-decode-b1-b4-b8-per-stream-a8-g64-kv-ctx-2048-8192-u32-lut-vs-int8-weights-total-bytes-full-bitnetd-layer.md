---
local: true
date: 2026-11-18
wave: "2026-11-18 14:55 (ran ~14:45–14:55 MT; Chief last_run 2026-11-18T21:10:00Z ~14:10 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-stream, continuous-batching, kv-cache, a8-g64, gqa, u32-lut, int8-control, ctx-8192, total-bytes, numpy, viable]
---

# Batching B independent decode streams through one u32-LUT weight pass is a real win: at ctx 2048, **B = 8 streams give 307–342 tok/s vs 45–48 tok/s for B = 1 (6.75–7.16x aggregate)** with total layer bytes (2-bit weights + 8 A8 g64 KV caches) of **37.9 MiB vs 87.6 MiB for int8 weights (2.31x smaller)**. At ctx 8192 B = 8 gives 4.13–4.25x aggregate tok/s, but the KV now dominates and the ternary total-bytes win falls to **1.49x** (101.7 vs 151.3 MiB), just under the bar. u32-LUT stays 0.65–1.00x the speed of int8 rows. Error unchanged (min per-stream layer-delta cos 1.0000000 vs f32-KV whole-softmax). Viable on memory for B ≤ 8 at ctx ≤ 2048 and B ≤ 4 at ctx 8192 (≥ 1.84x), with the batching speed-up on top.

## Question

[[2026-11-18_1055_split-k-lse-merge-flash-decoding-single-token-decode-a8-g64-kv-ctx-2048-8192-w8-pool-vs-per-head-flash1024-u32-lut-full-bitnetd-layer]] showed a B = 1 decode step is weight-expand bound (attention 7–17%) and asked whether the lever is fewer expanded bytes per step. Partial int8 tile caching was already measured in 2026-11-16_1455, so this run takes the other lever: amortise one expand pass over **several decode streams**. For B = 1 / 4 / 8 streams, each with its **own** A8 g64 KV cache at ctx 2048 / 8192, does the shared weight pass give ≥ 1.5x aggregate tok/s, and how much of the ternary 4x weight-memory win over int8 rows survives once B KV caches are counted?

Quant Chief `last_tips.json` `last_run` 2026-11-18T21:10:00Z (~14:10 MT): pair_A (Trit Pack & I2_S, Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock, Handshake & Ports) both **no gap**; roadmap blobs unchanged (4efb666 / bac6923). So this note uses the lane's own question. JuniorLLM tip at the push clone was 8e884e0 (10:46 MT, this lane's 10:55 note); no Trit Pack, Handshake or LLC commit newer, so nothing touched `junior_osai/vault/findings/` in the last 6 h.

## Method

- `base.py`, `par.py`, `tiled.py`, `cache.py` copied byte-identical from `/workspace/bench1055_1118`. Full bitnetd layer, hidden 2560, GQA 20 q / 5 KV heads × 128, FFN 6912, absmean ternary weights; fused q|k|v and gate|up. `OPENBLAS_NUM_THREADS=1`, one W = 8 thread pool.
- Weights: **u32** = 2-bit pack + u32-LUT expand (`par.fmm`, T = 256), 16.65 MiB; **int8** control = absmax int8 rows, same threading/tiling (`par.fmm8`), 66.34 MiB.
- 8 streams per weight set. Each stream has its own 8192-token Gaussian history projected through that weight set's qkv, stored as A8 g64 (int8 per token per 64-dim group + f32 scale) and as f32 (reference).
- Decode step for B streams at ctx n: X (B × 2560) → rms → qkv (one batched pass) → append each stream's k/v at position n−1 → attention as one pool task per (stream, KV head), online softmax over 1024-key tiles with per-tile dequant (the 10:55 `head1024` kernel) → o → gate|up → down. Every stream's `n` is reset after each step.
- Timing: 5 rounds × (2 warm + 8 timed) steps per config, rotating config order per round → 40 samples per config; median, min–max. Two full runs.
- Error: per-stream cos of the layer delta (y − x) vs the same weights with the f32 cache and an exact whole-context softmax.
- Bytes: total = weight bytes + B × A8 KV bytes at ctx n (2·5·n·128 int8 + 2·5·n·2 f32 scales).

Command: `cd /workspace/bench1455_1118 && OPENBLAS_NUM_THREADS=1 python3 mstream.py results_runN.json` (N = 1, 2).

<details><summary>mstream.py (inline)</summary>

```python
#!/usr/bin/env python3
"""Multi-stream (continuous-batched) decode, 14:55 11/18. B independent decode streams, each with its OWN A8 g64 KV cache
(int8 per token per 64-dim group + f32 scale) at ctx n, share one weight pass per step: X(B,D) -> rms -> fused qkv ->
per-stream append -> attention (one task per (stream, KV head), online softmax over 1024-key tiles, per-tile dequant,
W=8 pool) -> o -> gate|up -> down. Weights: 2-bit pack + u32-LUT expand (par.fmm, T=256, W=8, exact W1.58) vs the
int8-resident control (par.fmm8: absmax int8 rows, same threading/tiling, 4x the weight bytes).
Question: does sharing the weight pass across B streams give >=1.5x aggregate tok/s, and how much of the ternary 4x
weight-memory win survives once B streams of KV are counted (weights + B*KV bytes, u32 vs int8)?
base/par/tiled/cache byte-identical to /workspace/bench1055_1118. Usage: OPENBLAS_NUM_THREADS=1 python3 mstream.py out.json"""
import json, sys, time, statistics
import numpy as np
import base, par
from base import rms, silu, D, KVH, QH, HD
TU, W = 256, 8; G = QH // KVH; GS = 64; SC = np.float32(1 / np.sqrt(HD)); NMAX = 8192; TK = 1024
CTXS = (2048, 8192); BS = (1, 4, 8); ROUNDS, STEPS, WARM = 5, 8, 2
base.build(); u32_bytes = par.prep(); i8_bytes = par.prep8()
for n in base.PW:
    for f in ("Wd", "Wt", "wm", "ws", "wm8", "ws8", "PT", "q8"): base.PW[n].pop(f, None)
MM = {"u32": par.fmm, "int8": par.fmm8}; WB = {"u32": u32_bytes, "int8": i8_bytes}
def q8(x):
    g = x.reshape(x.shape[0], -1, HD // GS, GS); m = np.abs(g).max(-1) / 127.0 + 1e-12
    return np.clip(np.rint(g / m[..., None]), -127, 127).astype(np.int8).reshape(x.shape), m.astype(np.float32)
class KV:
    def __init__(s, k, v, fmt):   # k, v: (KVH, NMAX, HD) f32 history
        s.fmt = fmt; s.n = 0
        if fmt == "a8": (s.k, s.ks), (s.v, s.vs) = q8(k), q8(v)
        else: s.k, s.v = k.copy(), v.copy()
    def nbytes(s, n): return 2 * KVH * n * HD * s.k.itemsize + (2 * KVH * n * (HD // GS) * 4 if s.fmt == "a8" else 0)
    def append1(s, k, v):
        kh = k.reshape(KVH, 1, HD); vh = v.reshape(KVH, 1, HD); a = s.n
        if s.fmt == "a8": (s.k[:, a:a+1], s.ks[:, a:a+1]), (s.v[:, a:a+1], s.vs[:, a:a+1]) = q8(kh), q8(vh)
        else: s.k[:, a:a+1] = kh; s.v[:, a:a+1] = vh
        s.n = a + 1
    def get(s, h, a, b):
        if s.fmt == "f32": return s.k[h, a:b], s.v[h, a:b]
        n = b - a
        return ((s.k[h, a:b].reshape(n, -1, GS).astype(np.float32) * s.ks[h, a:b, :, None]).reshape(n, HD),
                (s.v[h, a:b].reshape(n, -1, GS).astype(np.float32) * s.vs[h, a:b, :, None]).reshape(n, HD))
def head(kv, Q, h):
    m = np.full((G, 1), -np.inf, np.float32); l = np.zeros((G, 1), np.float32); acc = np.zeros((G, HD), np.float32)
    for a in range(0, kv.n, TK):
        b = min(a + TK, kv.n); K, V = kv.get(h, a, b); S = Q @ K.T
        mn = np.maximum(m, S.max(1, keepdims=True)); al = np.exp(m - mn); S -= mn; np.exp(S, out=S)
        l *= al; l += S.sum(1, keepdims=True); acc *= al; acc += S @ V; m = mn
    return acc / l
def attn(kvs, q):
    B = q.shape[0]; qh = q.reshape(B, KVH, G, HD) * SC
    futs = [par.pool(W).submit(head, kvs[s], qh[s, h], h) for s in range(B) for h in range(KVH)]
    return np.stack([f.result() for f in futs]).reshape(B, D)
def attn_ref(kvs, q):  # exact: whole-context softmax over the f32 cache
    B = q.shape[0]; qh = q.reshape(B, KVH, G, HD) * SC; out = np.empty((B, KVH, G, HD), np.float32)
    for s in range(B):
        for h in range(KVH):
            K, V = kvs[s].get(h, 0, kvs[s].n); S = qh[s, h] @ K.T; S -= S.max(1, keepdims=True); np.exp(S, out=S)
            S /= S.sum(1, keepdims=True); out[s, h] = S @ V
    return out.reshape(B, D)
def step(mm, kvs, x, af):
    t0 = time.perf_counter(); h1 = rms(x); q, k, v = mm("qkv", h1, TU, W)
    for s in range(x.shape[0]): kvs[s].append1(k[s], v[s])
    t1 = time.perf_counter(); att = af(kvs, q); t2 = time.perf_counter()
    o = mm("o", att, TU, W)[0]; x2 = x + o; h2 = rms(x2)
    g, u = mm("gu", h2, TU, W); z = (silu(g) * u).astype(np.float32); y = x2 + mm("down", z, TU, W)[0]
    t3 = time.perf_counter(); return y, t2 - t1, t3 - t0
def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
def main():
    rng = np.random.default_rng(202611181455); BM = max(BS)
    XS = rng.standard_normal((STEPS, BM, D)).astype(np.float32)
    out = {"weight_bytes": WB, "cfg": {}}
    for wk, mm in MM.items():
        streams = []
        for s in range(BM):   # each stream: its own 8192-token Gaussian history through this weight set's qkv
            X0 = rng.standard_normal((NMAX, D)).astype(np.float32); K = np.empty((NMAX, KVH * HD), np.float32); V = np.empty_like(K)
            for c0 in range(0, NMAX, 512): _, K[c0:c0+512], V[c0:c0+512] = mm("qkv", rms(X0[c0:c0 + 512]), TU, W)
            K = K.reshape(NMAX, KVH, HD).transpose(1, 0, 2); V = V.reshape(NMAX, KVH, HD).transpose(1, 0, 2)
            streams.append((KV(K, V, "a8"), KV(K, V, "f32"))); del X0, K, V
        for n in CTXS:
            for B in BS:
                a8 = [streams[s][0] for s in range(B)]; f32 = [streams[s][1] for s in range(B)]
                for c in a8 + f32: c.n = n - 1
                ry, _, _ = step(mm, f32, XS[0, :B], attn_ref)
                for c in f32: c.n = n - 1
                y, _, _ = step(mm, a8, XS[0, :B], attn)
                for c in a8: c.n = n - 1
                d = dict(B=B, ctx=n, kv_bytes=B * a8[0].nbytes(n), attn_ms=[], step_ms=[],
                         layer_cos_min=min(cos(y[s] - XS[0, s], ry[s] - XS[0, s]) for s in range(B)), layer_maxabs=float(np.abs(y - ry).max()))
                out["cfg"][f"{wk}_ctx{n}_B{B}"] = d
        for n in CTXS:
            keys = [f"{wk}_ctx{n}_B{B}" for B in BS]
            for rd in range(ROUNDS):
                for kk in keys[rd % 3:] + keys[:rd % 3]:
                    d = out["cfg"][kk]; B = d["B"]; a8 = [streams[s][0] for s in range(B)]
                    for i in range(WARM + STEPS):
                        for c in a8: c.n = n - 1
                        _, ta, ts = step(mm, a8, XS[i % STEPS, :B], attn)
                        if i >= WARM: d["attn_ms"].append(ta * 1e3); d["step_ms"].append(ts * 1e3)
        del streams
    print(f"weight bytes u32 {WB['u32']/2**20:.2f} MiB, int8 {WB['int8']/2**20:.2f} MiB")
    print(f"{'cfg':18s} {'step ms med (min-max)':24s} {'attn ms med':11s} share  tok/s   agg_vs_B1  total_MiB  layer_cos_min")
    for d in out["cfg"].values(): d["step_med"] = statistics.median(d["step_ms"])
    for kk, d in out["cfg"].items():
        A, S_ = d["attn_ms"], d["step_ms"]; ma, ms = statistics.median(A), statistics.median(S_)
        b1 = out["cfg"][kk.rsplit("_B", 1)[0] + "_B1"]["step_med"]
        d.update(step_med=ms, step_min=min(S_), step_max=max(S_), attn_med=ma, share=ma / ms, tok_s=d["B"] * 1e3 / ms,
                 agg_vs_b1=d["B"] * b1 / ms, total_bytes=WB[kk.split("_")[0]] + d["kv_bytes"], n=len(S_))
        print(f"{kk:18s} {ms:7.2f} ({min(S_):.2f}-{max(S_):.2f})    {ma:7.2f}    {ma/ms:.3f}  {d['tok_s']:6.1f}  {d['agg_vs_b1']:5.2f}x    {d['total_bytes']/2**20:7.2f}   {d['layer_cos_min']:.7f}")
        del d["attn_ms"], d["step_ms"]
    print(f"{'u32 vs int8':18s} {'speed (int8_ms/u32_ms)':24s} {'bytes (int8/u32)':16s}")
    for n in CTXS:
        for B in BS:
            u, i = out["cfg"][f"u32_ctx{n}_B{B}"], out["cfg"][f"int8_ctx{n}_B{B}"]
            print(f"ctx{n}_B{B:<10d} {i['step_med']/u['step_med']:6.2f}x                  {i['total_bytes']/u['total_bytes']:5.2f}x")
            out.setdefault("u32_vs_int8", {})[f"ctx{n}_B{B}"] = dict(speed=i["step_med"] / u["step_med"], bytes=i["total_bytes"] / u["total_bytes"])
    json.dump(out, open(sys.argv[1], "w"), indent=1)
main()
```

</details>

## Results

Step = whole layer for all B streams. tok/s = B / step. agg = tok/s vs B = 1 at the same ctx and weights. Run 1 / run 2.

| weights | ctx | B | step ms med (min–max) run 1 | step ms med run 2 | attn ms med r1 / r2 | attn share r1 | tok/s r1 / r2 | agg vs B1 r1 / r2 | total MiB | layer cos min |
|---|---|---|---|---|---|---|---|---|---|---|
| u32 | 2048 | 1 | 20.96 (19.74–24.00) | 22.02 | 1.50 / 1.53 | 0.071 | 47.7 / 45.4 | 1.00x | 19.31 | 1.0000000 |
| u32 | 2048 | 4 | 19.81 (15.94–23.72) | 19.82 | 5.14 / 5.25 | 0.260 | 202.0 / 201.8 | 4.23x / 4.44x | 27.27 | 1.0000000 |
| u32 | 2048 | 8 | 23.42 (21.30–29.46) | 26.09 | 9.19 / 10.01 | 0.392 | 341.6 / 306.7 | **7.16x / 6.75x** | **37.90** | 1.0000000 |
| u32 | 8192 | 1 | 23.74 (21.98–26.30) | 24.96 | 4.00 / 4.29 | 0.168 | 42.1 / 40.1 | 1.00x | 27.27 | 1.0000000 |
| u32 | 8192 | 4 | 28.67 (27.23–32.09) | 30.26 | 15.18 / 15.81 | 0.530 | 139.5 / 132.2 | 3.31x / 3.30x | 59.15 | 1.0000000 |
| u32 | 8192 | 8 | 44.72 (41.96–73.23) | 48.33 | 29.13 / 31.86 | 0.651 | 178.9 / 165.5 | 4.25x / 4.13x | 101.65 | 1.0000000 |
| int8 | 2048 | 1 | 15.70 (14.89–19.50) | 15.70 | 1.70 / 1.62 | 0.108 | 63.7 / 63.7 | 1.00x | 68.99 | 0.9999999 |
| int8 | 2048 | 4 | 12.43 (11.74–16.94) | 12.95 | 4.68 / 4.80 | 0.377 | 321.9 / 308.8 | 5.05x / 4.85x | 76.96 | 0.9999998 |
| int8 | 2048 | 8 | 18.25 (16.81–23.89) | 18.69 | 8.92 / 9.08 | 0.489 | 438.4 / 428.1 | 6.88x / 6.72x | 87.59 | 0.9999998 |
| int8 | 8192 | 1 | 18.63 (17.53–21.42) | 18.10 | 4.32 / 4.32 | 0.232 | 53.7 / 55.2 | 1.00x | 76.96 | 1.0000000 |
| int8 | 8192 | 4 | 24.76 (22.78–27.99) | 27.30 | 15.51 / 17.43 | 0.627 | 161.6 / 146.5 | 3.01x / 2.65x | 108.84 | 0.9999999 |
| int8 | 8192 | 8 | 41.10 (38.04–47.48) | 48.15 | 30.04 / 35.66 | 0.731 | 194.6 / 166.2 | 3.63x / 3.01x | 151.34 | 0.9999999 |

Ternary (u32) vs int8 rows at the same ctx and B:

| ctx | B | speed int8_ms / u32_ms (r1 / r2) | total bytes int8 / u32 |
|---|---|---|---|
| 2048 | 1 | 0.75x / 0.71x | **3.57x** |
| 2048 | 4 | 0.63x / 0.65x | **2.82x** |
| 2048 | 8 | 0.78x / 0.72x | **2.31x** |
| 8192 | 1 | 0.78x / 0.73x | **2.82x** |
| 8192 | 4 | 0.86x / 0.90x | **1.84x** |
| 8192 | 8 | 0.92x / 1.00x | 1.49x |

Reading:
- One shared weight pass serves B streams almost for free at ctx 2048: 8 streams cost only 1.1–1.2x the B = 1 step. At ctx 8192 attention (B × 5 heads × 8192 keys) takes 52–66% of the step at B ≥ 4, so aggregate gain is capped at 3.3x (B = 4) and 4.1–4.3x (B = 8).
- As B × ctx grows, the KV bytes swamp the weights, so the ternary total-bytes edge drops from 3.57x to 1.49x, and the u32-LUT speed gap to int8 closes (0.65x → 0.92–1.00x) because attention, not weight expand, is now the step.
- Error is unchanged by batching (each stream's layer delta matches its f32-KV reference to cos ≥ 0.9999998).

## Verdict

**viable.** On the u32-LUT ternary path, multi-stream decode gives 4.1–7.2x aggregate tok/s over single-stream at ctx 2048–8192, and keeps ≥ 1.84x fewer total layer bytes than int8 rows up to B = 8 at ctx 2048 and B = 4 at ctx 8192 (2.31x at B = 8 / ctx 2048: 37.9 vs 87.6 MiB), at unchanged error. Not viable vs int8 on speed (0.63–1.00x). At B = 8 / ctx 8192 the ternary memory edge is 1.49x, just below the bar, because the A8 KV is now the bulk of the bytes. For i2sd: serve concurrent `juniorctl ask` streams batched through one weight pass; past ~B × ctx ≈ 16k tokens per layer the next memory lever is the KV, not the weights.

## Links

- previous: [[2026-11-18_1055_split-k-lse-merge-flash-decoding-single-token-decode-a8-g64-kv-ctx-2048-8192-w8-pool-vs-per-head-flash1024-u32-lut-full-bitnetd-layer]]
- related: [[2026-11-16_1455_partial-int8-tile-cache-hot-fused-sites-vs-threaded-u32-expand-and-all-int8-trit-tiles-45mb-budget-full-bitnetd-layer-b1-b8-b16]] (weight-only batching, no KV)
- this wave's pair files: none. Quant Chief 14:05 assignment was pair_A no gap and pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-18T21:10:00Z). Roadmap blobs: `JuniorHome docs/ROADMAP.md` 4efb666, `JuniorLLM docs/BETA_TO_OS.md` bac6923.
- bench dir: `/workspace/bench1455_1118` (mstream.py, run1.txt, run2.txt, results_run1.json, results_run2.json)

## Next question

At B = 8 / ctx 8192 the per-stream A8 g64 KV is ~65% of the bytes and of the step. Does a lower-bit KV for older tokens (A4 g32 nibble K/V beyond a recent 1024-token window kept at A8) restore a ≥ 1.5x ternary total-bytes edge over int8 weights and cut multi-stream attention time, at layer cos ≥ 0.9999 per stream?

close: sha256(prev || this)[:16] = e8119fe7d8ed91a8
