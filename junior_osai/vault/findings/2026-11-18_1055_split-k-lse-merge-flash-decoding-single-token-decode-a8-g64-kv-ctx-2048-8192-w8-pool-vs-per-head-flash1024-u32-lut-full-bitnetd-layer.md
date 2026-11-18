---
local: true
date: 2026-11-18
wave: "2026-11-18 10:55 (ran ~10:43–10:46 MT; Chief last_run 2026-11-18T17:05:00Z ~10:05 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, attention, flash-decoding, split-k, log-sum-exp, kv-cache, a8-g64, gqa, u32-lut, ctx-8192, threading, numpy, not-viable]
---

# Split-K does not help single-token decode on the W = 8 numpy pool: it is **slower** than one task per KV head. At ctx 8192 the per-head flash1024 kernel spends **3.99 / 3.87 ms** (runs 1 / 2) in attention; split2_1024 takes **4.60 / 4.50 ms (0.87x / 0.86x)**, split8_1024 **5.25 / 5.11 ms (0.76x)**, and finer 256- or 128-key tiles are 0.17–0.38x. At ctx 2048 split-K is 0.77–0.81x. Error is identical (attention cos 0.9999173 at 8192, layer-delta cos 0.9999999). Attention is only **17% of the u32-LUT decode step at ctx 8192 (3.9 of 22.7–23.3 ms) and 7% at ctx 2048**, so even a perfect attention speed-up would make the step at most 1.2x faster. Far below the 1.5x bar → not-viable.

## Question

[[2026-11-18_0555_split-k-lse-merge-and-q-head-split-attention-w8-pool-a8-g64-kv-chunked-prefill-ctx-8192-vs-per-head-flash1024-u32-lut-full-bitnetd-layer]] found split-K gave only 1.2–1.4x on prefill attention. Its next question, run here: for a single-token decode step (B = 1, 4 q rows per KV head) over an A8 g64 KV cache of 2048 / 8192 tokens, does split-K across the W = 8 pool with log-sum-exp merge make attention ≥ 1.5x faster than one task per KV head, and what share of the full u32-LUT layer decode step does attention take?

Quant Chief `last_tips.json` `last_run` 2026-11-18T17:05:00Z (~10:05 MT): pair_A (Trit Pack & I2_S, Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock, Handshake & Ports) both **no gap**; roadmap blobs unchanged (4efb666 / bac6923). So this note uses the lane's own question. JuniorLLM tip at the push clone was 162fe06 (05:53 MT, this lane's 05:55 note); no Trit Pack, Handshake or LLC commit newer than it, so none touched `junior_osai/vault/findings/` in the last 6 h.

Prior findings scanned (154 notes): 2026-11-10_1455 split decode *stages* (dequant vs score), not keys across workers; no earlier note runs split-K for a one-token decode.

## Method

- `base.py`, `par.py`, `tiled.py`, `cache.py` byte-identical to `/workspace/bench0555_1118` (cmp verified). Full bitnetd layer, hidden 2560, GQA 20 q / 5 KV heads × 128, FFN 6912, absmean ternary weights, fused q|k|v and gate|up; weights = 2-bit pack + u32-LUT expand (`par.fmm`, T = 256, W = 8), `OPENBLAS_NUM_THREADS=1`. One W = 8 thread pool shared by weight GEMVs and attention.
- KV history: 8192 Gaussian tokens (seed 202611181055) through rms + the same qkv weights; stored f32 (reference), f16 and A8 g64 (int8 per token per 64-dim group + f32 scale).
- Decode step at ctx n: new token → rms → qkv → append K/V at position n−1 → attention over all n keys → o → FFN. `kv.n` is reset after every step, so each step sees the same ctx. Attention time = from the end of append to the end of attention (including the LSE merge); step time = the whole layer.
- Attention kernels (online softmax over T-key tiles, per-tile dequant):
  - `head1024`: baseline, one task per KV head (5 tasks on 8 workers); `f16_head1024` same on the f16 cache; `serial_head1024_w1` same on one thread; `head256` with 256-key tiles.
  - `splitS_T`: each head's tiles cut into S contiguous groups (`np.array_split`, S capped at the tile count) → 5·S tasks, each returns (m, l, acc), merged per head: M = max m_j, out = Σ e^(m_j−M) acc_j / Σ e^(m_j−M) l_j. S/T ∈ {2/1024, 8/1024, 8/256, 16/256, 32/128}.
- Timing: per ctx, 5 rounds (path order rotated) × (3 warm + 16 timed steps) per path → 80 timed steps per path; median (min–max). Two full runs (50 s and 49 s wall; load avg 0.9 and 2.6 before them).
- Error: attention output and layer delta vs the f32 cache + whole-context exact softmax, same new token.

Commands (in `/workspace/bench1055_1118`, `decode.py` sha256 `a428e3d12335b6e4…`):

```
OPENBLAS_NUM_THREADS=1 python3 decode.py results_run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 python3 decode.py results_run2.json > run2.txt
```

<details><summary>decode.py</summary>

```python
#!/usr/bin/env python3
"""Split-K (flash-decoding, LSE merge) vs one-task-per-KV-head attention for a single-token DECODE step (10:55 11/18).
Full bitnetd layer at B=1 (hidden 2560, 20 q / 5 KV heads x 128, FFN 6912), weights = 2-bit pack + u32-LUT expand
(par.fmm, T=256, W=8, exact W1.58). KV history = k/v of 8192 Gaussian tokens projected through the same qkv weights,
stored A8 g64 (int8 per token per 64-dim group + f32 scale) or f16; f32 copy = exact reference.
Decode step at ctx n: q,k,v of the new token; append at pos n-1; attention over all n keys (no mask); o; FFN.
After each step kv.n is reset so every step sees the same ctx.
  head_T   : one task per KV head (4 q rows), online softmax over T-key tiles, per-tile dequant -> 5 tasks on W=8
  splitS_T : each head's tiles cut into S contiguous groups -> 5*S tasks, each returns (m, l, acc), LSE merge per head
  serial   : head_1024 on 1 thread (W=1)
base/par/tiled/cache byte-identical to /workspace/bench0555_1118. Usage: OPENBLAS_NUM_THREADS=1 python3 decode.py out.json"""
import json, sys, time, statistics
import numpy as np
import base, par
from base import rms, silu, D, KVH, QH, HD
TU, W = 256, 8; G = QH // KVH; GS = 64; SC = np.float32(1 / np.sqrt(HD)); NMAX = 8192
CTXS = (2048, 8192); ROUNDS, STEPS, WARM = 5, 16, 3
base.build(); u32_bytes = par.prep()
for n in base.PW:
    for f in ("Wd", "Wt", "wm", "ws", "wm8", "ws8", "PT", "q8"): base.PW[n].pop(f, None)
class KV:
    def __init__(s, fmt):
        s.fmt = fmt; dt = {"f32": np.float32, "f16": np.float16, "a8": np.int8}[fmt]
        s.k = np.zeros((KVH, NMAX, HD), dt); s.v = np.zeros((KVH, NMAX, HD), dt); s.n = 0
        if fmt == "a8": s.ks = np.zeros((KVH, NMAX, HD // GS), np.float32); s.vs = np.zeros((KVH, NMAX, HD // GS), np.float32)
    def nbytes(s, n): return 2 * KVH * n * HD * s.k.itemsize + (2 * KVH * n * (HD // GS) * 4 if s.fmt == "a8" else 0)
    def _q8(s, x):
        g = x.reshape(KVH, -1, HD // GS, GS); m = np.abs(g).max(-1) / 127.0 + 1e-12
        return np.clip(np.rint(g / m[..., None]), -127, 127).astype(np.int8).reshape(x.shape), m.astype(np.float32)
    def append(s, k, v):
        c = k.shape[0]; a, b = s.n, s.n + c
        kh = k.reshape(c, KVH, HD).transpose(1, 0, 2); vh = v.reshape(c, KVH, HD).transpose(1, 0, 2)
        if s.fmt == "a8": s.k[:, a:b], s.ks[:, a:b] = s._q8(kh); s.v[:, a:b], s.vs[:, a:b] = s._q8(vh)
        else: s.k[:, a:b] = kh; s.v[:, a:b] = vh
        s.n = b
    def get(s, h, a, b):
        if s.fmt == "f32": return s.k[h, a:b], s.v[h, a:b]
        if s.fmt == "f16": return s.k[h, a:b].astype(np.float32), s.v[h, a:b].astype(np.float32)
        n = b - a
        kk = (s.k[h, a:b].reshape(n, -1, GS).astype(np.float32) * s.ks[h, a:b, :, None]).reshape(n, HD)
        vv = (s.v[h, a:b].reshape(n, -1, GS).astype(np.float32) * s.vs[h, a:b, :, None]).reshape(n, HD)
        return kk, vv
def part(kv, Q, h, a0, a1, T):
    m = np.full((G, 1), -np.inf, np.float32); l = np.zeros((G, 1), np.float32); acc = np.zeros((G, HD), np.float32)
    for a in range(a0, a1, T):
        b = min(a + T, a1); K, V = kv.get(h, a, b); S = Q @ K.T
        mn = np.maximum(m, S.max(1, keepdims=True)); al = np.exp(m - mn); S -= mn; np.exp(S, out=S)
        l *= al; l += S.sum(1, keepdims=True); acc *= al; acc += S @ V; m = mn
    return m, l, acc
def attn_whole(kv, q):  # exact reference: whole context, one softmax
    out = np.empty((QH, HD), np.float32); qh = q.reshape(KVH, G, HD) * SC
    for h in range(KVH):
        K, V = kv.get(h, 0, kv.n); S = qh[h] @ K.T; S -= S.max(1, keepdims=True); np.exp(S, out=S); S /= S.sum(1, keepdims=True)
        out[h * G:(h + 1) * G] = S @ V
    return out.reshape(1, D)
def attn_split(kv, q, T, S, Wn):
    qh = q.reshape(KVH, G, HD) * SC; n = kv.n; nt = -(-n // T); S = min(S, nt)
    bounds = [(g[0] * T, min((g[-1] + 1) * T, n)) for g in np.array_split(np.arange(nt), S)]
    tasks = [(h, a, b) for h in range(KVH) for (a, b) in bounds]
    if Wn == 1: res = [part(kv, qh[h], h, a, b, T) for (h, a, b) in tasks]
    else: res = [f.result() for f in [par.pool(Wn).submit(part, kv, qh[h], h, a, b, T) for (h, a, b) in tasks]]
    out = np.empty((QH, HD), np.float32)
    for h in range(KVH):
        rs = res[h * S:(h + 1) * S]
        if S == 1: m, l, acc = rs[0]; out[h * G:(h + 1) * G] = acc / l; continue
        M = np.max(np.concatenate([r[0] for r in rs], 1), 1, keepdims=True)
        L = sum(np.exp(r[0] - M) * r[1] for r in rs); A = sum(np.exp(r[0] - M) * r[2] for r in rs)
        out[h * G:(h + 1) * G] = A / L
    return out.reshape(1, D)
PATHS = {"serial_head1024_w1": ("a8", lambda kv, q: attn_split(kv, q, 1024, 1, 1)),
         "f16_head1024": ("f16", lambda kv, q: attn_split(kv, q, 1024, 1, W)),
         "head1024": ("a8", lambda kv, q: attn_split(kv, q, 1024, 1, W)),
         "head256": ("a8", lambda kv, q: attn_split(kv, q, 256, 1, W)),
         "split2_1024": ("a8", lambda kv, q: attn_split(kv, q, 1024, 2, W)),
         "split8_1024": ("a8", lambda kv, q: attn_split(kv, q, 1024, 8, W)),
         "split8_256": ("a8", lambda kv, q: attn_split(kv, q, 256, 8, W)),
         "split16_256": ("a8", lambda kv, q: attn_split(kv, q, 256, 16, W)),
         "split32_128": ("a8", lambda kv, q: attn_split(kv, q, 128, 32, W))}
def step(kv, x, attn):
    t0 = time.perf_counter(); h1 = rms(x); q, k, v = par.fmm("qkv", h1, TU, W); kv.append(k, v)
    t1 = time.perf_counter(); att = attn(kv, q); t2 = time.perf_counter()
    o = par.fmm("o", att, TU, W)[0]; x2 = x + o; h2 = rms(x2)
    g, u = par.fmm("gu", h2, TU, W); z = (silu(g) * u).astype(np.float32); y = x2 + par.fmm("down", z, TU, W)[0]
    t3 = time.perf_counter(); return y, att, t2 - t1, t3 - t0
def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
def main():
    rng = np.random.default_rng(202611181055)
    X0 = rng.standard_normal((NMAX, D)).astype(np.float32); XS = rng.standard_normal((STEPS, D)).astype(np.float32)
    caches = {f: KV(f) for f in ("f32", "f16", "a8")}
    for c0 in range(0, NMAX, 512):
        _, k, v = par.fmm("qkv", rms(X0[c0:c0 + 512]), TU, W)
        for kv in caches.values(): kv.append(k, v)
    out = {"ctx": {}, "u32_weight_bytes": u32_bytes}
    for n in CTXS:
        r = {"kv_bytes": {f: caches[f].nbytes(n) for f in ("f32", "f16", "a8")}, "paths": {}}
        for kv in caches.values(): kv.n = n - 1
        refy, refa, _, _ = step(caches["f32"], XS[:1], attn_whole); caches["f32"].n = n - 1
        for p, (f, fn) in PATHS.items():
            kv = caches[f]; y, a, _, _ = step(kv, XS[:1], fn); kv.n = n - 1
            r["paths"][p] = {"attn_ms": [], "step_ms": [], "attn_maxabs": float(np.abs(a - refa).max()), "attn_cos": cos(a, refa),
                             "layer_maxabs": float(np.abs(y - refy).max()), "layer_cos": cos(y - XS[:1], refy - XS[:1])}
        names = list(PATHS)
        for rd in range(ROUNDS):
            for p in names[rd % len(names):] + names[:rd % len(names)]:
                f, fn = PATHS[p]; kv = caches[f]
                for i in range(WARM + STEPS):
                    _, _, ta, ts = step(kv, XS[i % STEPS:i % STEPS + 1], fn); kv.n = n - 1
                    if i >= WARM: r["paths"][p]["attn_ms"].append(ta * 1e3); r["paths"][p]["step_ms"].append(ts * 1e3)
        b = statistics.median(r["paths"]["head1024"]["attn_ms"])
        print(f"ctx {n}: KV bytes f32 {r['kv_bytes']['f32']/2**20:.2f} f16 {r['kv_bytes']['f16']/2**20:.2f} a8 {r['kv_bytes']['a8']/2**20:.2f} MiB")
        print(f"{'path':22s} {'attn ms med (min-max)':26s} {'step ms med (min-max)':26s} share  vs_head1024  attn_cos   layer_cos  maxabs")
        for p, d in r["paths"].items():
            A, S_ = d["attn_ms"], d["step_ms"]; ma, ms = statistics.median(A), statistics.median(S_)
            d.update(attn_med=ma, step_med=ms, share=ma / ms, speedup_attn=b / ma)
            print(f"{p:22s} {ma:7.3f} ({min(A):.3f}-{max(A):.3f})      {ms:7.3f} ({min(S_):.3f}-{max(S_):.3f})     {ma/ms:.3f}  {b/ma:6.2f}x   {d['attn_cos']:.7f} {d['layer_cos']:.7f} {d['attn_maxabs']:.2e}")
            del d["attn_ms"], d["step_ms"]
        out["ctx"][n] = r
    json.dump(out, open(sys.argv[1], "w"), indent=1)
main()
```
</details>

## Results

KV bytes for 5 KV heads × 128 × (K + V): ctx 2048 f32 10.00 / f16 5.00 / A8 2.66 MiB; ctx 8192 f32 40.00 / f16 20.00 / A8 10.62 MiB (A8 vs f16 = 1.88x fewer bytes, as in 02:55). Split-K adds no resident bytes (merge partials 5·S·4·130 floats, < 0.1 MiB). Weights 16.6 MiB/layer (u32 2-bit pack).

### ctx 8192 (attention ms and step ms: median (min–max) over 80 steps; run 1 / run 2)

| path | attn ms run 1 | attn ms run 2 | step ms run 1 | step ms run 2 | attn share | vs head1024 | attn cos | layer cos |
|---|---|---|---|---|---|---|---|---|
| serial_head1024_w1 | 8.891 (8.245–12.203) | 8.923 (8.338–10.329) | 28.653 | 28.485 | 0.31 | 0.45x / 0.43x | 0.9999173 | 0.9999999 |
| f16_head1024 | 3.987 (3.531–6.841) | 4.026 (3.540–7.105) | 22.704 | 23.175 | 0.17–0.18 | 1.00x / 0.96x | 0.9999322 | 1.0000000 |
| **head1024 (baseline)** | **3.987 (3.487–6.411)** | **3.872 (3.465–4.977)** | **23.261** | **22.669** | **0.17** | 1.00x | 0.9999173 | 0.9999999 |
| head256 | 10.450 | 10.397 | 29.741 | 29.279 | 0.35 | 0.38x / 0.37x | 0.9999173 | 0.9999999 |
| split2_1024 | 4.601 (4.251–6.876) | 4.499 (4.207–4.969) | 23.401 | 23.368 | 0.19–0.20 | 0.87x / 0.86x | 0.9999173 | 0.9999999 |
| split8_1024 | 5.249 (4.778–5.976) | 5.113 (4.712–6.887) | 24.328 | 23.840 | 0.21–0.22 | 0.76x / 0.76x | 0.9999173 | 0.9999999 |
| split8_256 | 12.507 | 11.925 | 31.807 | 30.721 | 0.39 | 0.32x / 0.32x | 0.9999173 | 0.9999999 |
| split16_256 | 13.392 | 12.774 | 32.549 | 31.560 | 0.41 | 0.30x / 0.30x | 0.9999173 | 0.9999999 |
| split32_128 | 23.452 | 22.501 | 42.350 | 41.500 | 0.54–0.55 | 0.17x / 0.17x | 0.9999173 | 0.9999999 |

### ctx 2048

| path | attn ms run 1 | attn ms run 2 | step ms run 1 | step ms run 2 | attn share | vs head1024 |
|---|---|---|---|---|---|---|
| serial_head1024_w1 | 2.284 | 2.269 | 21.611 | 21.525 | 0.11 | 0.64x / 0.65x |
| f16_head1024 | 1.505 | 1.456 | 21.278 | 20.569 | 0.07 | 0.97x / 1.02x |
| **head1024 (baseline)** | **1.454 (1.269–2.194)** | **1.480 (1.290–3.745)** | **20.620** | **20.784** | **0.07** | 1.00x |
| head256 | 3.228 | 3.094 | 23.546 | 22.076 | 0.14 | 0.45x / 0.48x |
| split2_1024 | 1.885 | 1.835 | 21.546 | 20.424 | 0.09 | 0.77x / 0.81x |
| split8_1024 | 1.891 | 1.823 | 21.287 | 20.448 | 0.09 | 0.77x / 0.81x |
| split8_256 | 4.200 | 4.096 | 22.869 | 22.431 | 0.18 | 0.35x / 0.36x |
| split16_256 | 4.276 | 4.072 | 23.001 | 22.445 | 0.18–0.19 | 0.34x / 0.36x |
| split32_128 | 7.171 | 7.026 | 26.394 | 25.846 | 0.27 | 0.20x / 0.21x |

Attention max abs error vs exact: 3.91e-4 (ctx 2048) and 5.08e-4 (ctx 8192) for every A8 path. Splitting does not change the error.

Readings:
- With 4 query rows per head, each tile does very little math. Time is spent on per-tile Python/numpy dispatch (dequant, two small matmuls, exp, rescale), and every extra task adds submit/result overhead. More tasks or smaller tiles means more fixed cost: time per path tracks the number of tiles, not the parallel width.
- One thread vs 5 tasks on 8 workers is only 2.2–2.3x at ctx 8192 (8.9 → 3.9 ms), so the per-head kernel is already limited by the GIL or memory, not by idle workers. Adding the 3 idle workers through split-K cannot recover more.
- f16 and A8 run at the same decode speed (0.96–1.02x), so A8 keeps its 1.88x KV-memory win at no decode cost. This confirms the 02:55 memory finding and does not count as a new speed gain.
- The weights dominate the decode step (≈ 19 ms of ≈ 23 ms at ctx 8192). Attention at 17% caps any attention-only gain at ≤ 1.2x per step.

## Verdict

**not-viable.** Split-K / flash-decoding with LSE merge is 0.76–0.87x (slower) for one-token decode at ctx 2048 and 8192 on this numpy W = 8 pool, at identical error. Keep one task per KV head with 1024-key tiles (`head1024`) on the A8 g64 cache for decode.

## Links

- previous: [[2026-11-18_0555_split-k-lse-merge-and-q-head-split-attention-w8-pool-a8-g64-kv-chunked-prefill-ctx-8192-vs-per-head-flash1024-u32-lut-full-bitnetd-layer]]
- this wave's pair files: none. Quant Chief 10:05 assignment was pair_A no gap and pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-18T17:05:00Z). Roadmap blobs: `JuniorHome docs/ROADMAP.md` 4efb666, `JuniorLLM docs/BETA_TO_OS.md` bac6923.
- bench dir: `/workspace/bench1055_1118` (decode.py, run1.txt, run2.txt, results_run1.json, results_run2.json)

## Next question

Attention is only 7–17% of a decode step, and the u32-LUT weight GEMVs take about 19 ms of it. At B = 1 decode over ctx 8192, does running attention for head h *concurrently* with the o-projection rows that don't depend on it (or overlapping the next layer's qkv expand with the FFN down) on the same W = 8 pool hide attention entirely? Or is the step bound by the weight-expand bytes, so that the real lever is fewer expanded bytes per step (for example caching hot int8 tiles for o / down only) at ctx 8192?

close: sha256(prev || this)[:16] = e5bd1db144259435
