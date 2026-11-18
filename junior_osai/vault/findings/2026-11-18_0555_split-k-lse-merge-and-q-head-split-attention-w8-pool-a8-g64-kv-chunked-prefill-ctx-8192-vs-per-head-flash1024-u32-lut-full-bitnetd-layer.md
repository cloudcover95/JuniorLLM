---
local: true
date: 2026-11-18
wave: "2026-11-18 05:55 (ran ~05:48–05:55 MT; Chief last_run 2026-11-18T12:05:00Z ~05:05 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (long-prompt prefill)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, prefill, attention, flash-attention, flash-decoding, split-k, log-sum-exp, kv-cache, a8-g64, gqa, u32-lut, ctx-8192, threading, numpy, not-viable]
---

# Filling the 3 idle workers helps attention only 1.2–1.4x: in the same 8192-token chunked prefill (C = 512, 2-bit u32-LUT weights, A8 g64 KV), **split-K with log-sum-exp merge (8 key groups per head, 40 tasks)** spends **627 / 661 ms in attention vs 791 / 912 ms for the 02:55 per-head flash1024 kernel (1.26x / 1.38x, runs 1 / 2)**; a simpler **per-q-head split (20 tasks, no merge)** is the same, **636 / 652 ms (1.24x / 1.40x)**. Error is unchanged (max abs 1.37e-2, min cos 0.999997; split vs per-head ≤ 9.5e-7). Last chunk (ctx 8192) attention 79 / 90 → 63 / 69 ms; share at 8192 drops 0.45 → 0.38–0.40, **not ≤ 0.35**. Whole prefill only 1.08x / 1.15x faster. KV bytes unchanged (10.6 MiB); split-K adds up to 40.6 MiB of transient merge partials. Below the 1.5x bar → not-viable.

## Question

[[2026-11-18_0255_flash-tiled-online-softmax-attention-per-tile-a8-g64-kv-dequant-vs-whole-context-f16-f32-a8-chunked-prefill-ctx-8192-u32-lut-full-bitnetd-layer]] made attention linear in context with 1024-key tiles and per-tile A8 dequant, but only 5 KV heads feed the W = 8 pool, so 3 workers idle during attention. Its next question, run here: does splitting each head's key tiles across workers (flash-decoding split-KV, per-split m / l / acc merged by log-sum-exp) bring the ctx-8192 attention share from ~0.43 to ≤ 0.35 (≥ 1.3x attention speed-up) at the same error? Saved viability bar is 1.5x on speed or memory.

Quant Chief `last_tips.json` `last_run` 2026-11-18T12:05:00Z (~05:05 MT): pair_A (Trit Pack & I2_S, Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock, Handshake & Ports) both **no gap** ("05:05 11/18: blobs unchanged; both no gap"); roadmap blobs unchanged (4efb666 / bac6923). So this note uses the lane's own question. JuniorLLM tip at the push clone was 409eaa6 (02:56 MT, this lane's 02:55 note); prior commit fd72daf (02:49 MT, `rails/livefalse` only). No Trit Pack, Handshake or LLC commit touched `junior_osai/vault/findings/` in the last 6 h.

Prior findings scanned (153 notes, `ls junior_osai/vault/findings`): no earlier note covers split-K / LSE-merged or q-head-split attention parallelism; 2026-11-10_1455 split decode *stages*, not keys across workers.

## Method

- **Unchanged:** `base.py`, `tiled.py`, `par.py`, `cache.py` byte-identical to `/workspace/bench0255_1118` (sha256 verified): hidden 2560, GQA 20 q / 5 KV heads × 128, FFN 6912, absmean ternary weights, fused q|k|v and gate|up, `OPENBLAS_NUM_THREADS=1`, W = 8 worker pool shared by weight GEMMs and attention. Weights = 2-bit pack + u32-LUT expand (`par.fmm`, T = 256), 16.6 MiB/layer. KV = A8 g64 (int8 per token per 64-dim group + f32 scale), 10.6 MiB at 8192 tokens.
- **Kernels** (all: 1024-key tiles unless noted, per-tile A8 dequant, causal mask only on diagonal-crossing tiles, online softmax):
  - `flash1024` — 02:55 baseline: one task per KV head (its 4 q heads stacked, 2048 rows) → 5 tasks.
  - `splitS_T` — each head's tiles cut into S contiguous groups (`np.array_split`, S capped by tile count) → 5·S tasks; each returns unnormalized (m, l, acc), rows with no visible key keep m = −inf (safe shift so exp gives 0, not NaN); per head merged by LSE: M = max m_j, out = Σ e^(m_j−M) acc_j / Σ e^(m_j−M) l_j. S ∈ {2, 4, 8} at T = 1024, plus S = 8 at T = 512.
  - `qsplit4_1024` — one task per q head (20 tasks, 512 rows each), each re-dequantizes its KV head's tiles (4x dequant work, no merge).
- **Prefill:** P = 8192 Gaussian tokens (seed 202611180555) in 16 chunks of C = 512; per chunk weight time and attention time (KV append/quant + dequant + scores + softmax + PV + merge) timed separately.
- **Timing:** 1 error pass per path, then 4 rounds (rotated order) × 1 full prefill per path; median (min–max). Two full runs (90 s and 97 s wall). Load avg 0.7 before run 1, 2.7 before run 2 (run-2 baseline noisier: attn 801–958 ms).
- **Error:** whole 8192-token output vs `f32 + whole-context` exact reference; also max abs vs the a8 + flash1024 output.
- **Diag (`diag.py`):** last chunk only (ctx 8192, synthetic A8 KV), attention alone, 9 reps: serial 1-thread vs W = 8 kernels, to get parallel efficiency.

Commands (in `/workspace/bench0555_1118`, `splitkv.py` sha256 `f18c0bd52a05b352…`, `diag.py` sha256 `a3b80713df88e83e…`):

```
OPENBLAS_NUM_THREADS=1 python3 splitkv.py results_run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 python3 splitkv.py results_run2.json > run2.txt
OPENBLAS_NUM_THREADS=1 python3 diag.py > diag.txt
```

<details><summary>splitkv.py</summary>

```python
#!/usr/bin/env python3
"""Split-K (flash-decoding style) vs per-head flash attention on the W = 8 pool, A8 g64 KV (05:55 11/18).
Chunked prefill P = 8192 tokens in C = 512 chunks through the full bitnetd layer (hidden 2560, 20 q / 5 KV heads x 128,
FFN 6912), weights = 2-bit pack + u32-LUT expand (par.fmm, T=256, W=8; exact W1.58, the 14:55/18:55 pick).
KV: f32 / f16 / A8 g64 (int8 per token per 64-dim group, f32 scale). Attention per KV head (4 q heads stacked), causal.
  whole : 18:55 kernel - dequant/cast the whole context, one S = QK^T over all keys, boolean causal mask, softmax, @V.
  flashT: key tiles of T tokens; only tiles <= last query pos; dequant/cast ONE tile, S tile, causal mask only on tiles
          that cross the chunk diagonal, running max m / sum l / acc rescale (online softmax), divide at the end.
02:55 baseline: flash1024 = one task per KV head (5 tasks on 8 workers -> 3 idle).
New: splitS_T = each head's key tiles cut into S contiguous groups, one task per (head, group) -> 5*S tasks; each task
     returns unnormalized (m, l, acc); per head merged by log-sum-exp: M = max m_j, out = sum e^(m_j-M) acc_j / sum e^(m_j-M) l_j.
     qsplit4 = one task per q head (20 tasks), each re-dequantizes its head's tiles (no merge, 4x dequant work).
Weight GEMM time and attention time per chunk are timed separately. base/par/tiled/cache byte-identical to
/workspace/bench0255_1118. Usage: OPENBLAS_NUM_THREADS=1 python3 splitkv.py out.json"""
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

def flash_part(kv, Q, qpos, h, c0, tiles, T):
    """online softmax over the given key tiles; returns unnormalized m, l, acc. Rows with no visible key keep m=-inf."""
    R = Q.shape[0]; n = kv.n
    m = np.full((R, 1), -np.inf, np.float32); l = np.zeros((R, 1), np.float32); acc = np.zeros((R, HD), np.float32)
    for t in tiles:
        a = t * T; b = min(a + T, n); K, V = kv.get(h, a, b)
        S = Q @ K.T
        if b - 1 > c0: S[np.arange(a, b)[None, :] > qpos] = -np.inf
        mn = np.maximum(m, S.max(1, keepdims=True)); sf = np.where(np.isfinite(mn), mn, np.float32(0))
        al = np.exp(m - sf); S -= sf; np.exp(S, out=S)
        l *= al; l += S.sum(1, keepdims=True); acc *= al; acc += S @ V; m = mn
    return m, l, acc
def attn_flash(kv, qh, c0, out, T):
    c = qh.shape[2]; nt = -(-kv.n // T)
    def task(h):
        Q = qh[h].reshape(G * c, HD) * SC; qpos = np.tile(c0 + np.arange(c), G)[:, None]
        m, l, acc = flash_part(kv, Q, qpos, h, c0, range(nt), T); acc /= l
        out[:, h] = acc.reshape(G, c, HD).transpose(1, 0, 2)
    for f in [par.pool(W).submit(task, h) for h in range(KVH)]: f.result()
def attn_split(kv, qh, c0, out, T, S):
    c = qh.shape[2]; nt = -(-kv.n // T); groups = [g for g in np.array_split(np.arange(nt), min(S, nt)) if len(g)]
    Qs = [qh[h].reshape(G * c, HD) * SC for h in range(KVH)]; qpos = np.tile(c0 + np.arange(c), G)[:, None]
    futs = {(h, j): par.pool(W).submit(flash_part, kv, Qs[h], qpos, h, c0, g, T) for h in range(KVH) for j, g in enumerate(groups)}
    for h in range(KVH):
        parts = [futs[(h, j)].result() for j in range(len(groups))]
        M = parts[0][0].copy()
        for p in parts[1:]: np.maximum(M, p[0], out=M)
        num = np.zeros((G * c, HD), np.float32); den = np.zeros((G * c, 1), np.float32)
        for m, l, acc in parts:
            w = np.exp(m - M); num += w * acc; den += w * l
        num /= den; out[:, h] = num.reshape(G, c, HD).transpose(1, 0, 2)
    STAT["merge_bytes"] = max(STAT.get("merge_bytes", 0), KVH * len(groups) * G * c * (HD + 2) * 4)
def attn_qsplit(kv, qh, c0, out, T):
    c = qh.shape[2]; nt = -(-kv.n // T); qpos = (c0 + np.arange(c))[:, None]
    def task(h, g):
        Q = qh[h, g] * SC; m, l, acc = flash_part(kv, Q, qpos, h, c0, range(nt), T); acc /= l
        out[:, h, g] = acc
    for f in [par.pool(W).submit(task, h, g) for h in range(KVH) for g in range(G)]: f.result()
STAT = {}
KERN = {"flash1024": lambda kv, qh, c0, out: attn_flash(kv, qh, c0, out, 1024),
        "split2_1024": lambda kv, qh, c0, out: attn_split(kv, qh, c0, out, 1024, 2),
        "split4_1024": lambda kv, qh, c0, out: attn_split(kv, qh, c0, out, 1024, 4),
        "split8_1024": lambda kv, qh, c0, out: attn_split(kv, qh, c0, out, 1024, 8),
        "split8_512": lambda kv, qh, c0, out: attn_split(kv, qh, c0, out, 512, 8),
        "qsplit4_1024": lambda kv, qh, c0, out: attn_qsplit(kv, qh, c0, out, 1024)}
def head_whole(kv, qh, h, c0, out):
    K, V = kv.get(h, 0, kv.n); n = K.shape[0]; c = qh.shape[1]
    S = (qh.reshape(G * c, HD) @ K.T).reshape(G, c, n); S *= SC
    pos = c0 + np.arange(c)[:, None]; S[:, np.arange(n)[None, :] > pos] = -np.inf
    S -= S.max(-1, keepdims=True); np.exp(S, out=S); S /= S.sum(-1, keepdims=True)
    out[:, h] = (S.reshape(G * c, n) @ V).reshape(G, c, HD).transpose(1, 0, 2)
def attn_whole(kv, qh, c0, out):
    for f in [par.pool(W).submit(head_whole, kv, qh[h], h, c0, out) for h in range(KVH)]: f.result()
KERN["whole"] = attn_whole
def attn(kv, kern, q, k, v, c0):
    c = q.shape[0]; kv.append(k, v)
    qh = q.reshape(c, KVH, G, HD).transpose(1, 2, 0, 3)
    out = np.empty((c, KVH, G, HD), np.float32)
    KERN[kern](kv, qh, c0, out)
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
PATHS = [("a8", "flash1024"), ("a8", "split2_1024"), ("a8", "split4_1024"), ("a8", "split8_1024"),
         ("a8", "split8_512"), ("a8", "qsplit4_1024")]
ROUNDS = 4
def main():
    out = sys.argv[1]; rng = np.random.default_rng(202611180555)
    X = rng.standard_normal((P_TOK, D)).astype(np.float32)
    ref, _ = prefill("f32", "whole", X)
    yb, _ = prefill("a8", "flash1024", X)
    res = {"u32_weight_bytes": u32_bytes, "P": P_TOK, "C": C, "rounds": ROUNDS, "paths": {}}
    for p in PATHS:
        y, kv = prefill(*p, X)
        d = (y - ref).astype(np.float64); r = ref.astype(np.float64)
        cs = np.sum(y * r, 1) / (np.linalg.norm(y, axis=1) * np.linalg.norm(r, axis=1))
        res["paths"]["%s+%s" % p] = dict(max_abs=float(np.abs(d).max()), rel_fro=float(np.linalg.norm(d) / np.linalg.norm(r)),
                                         min_cos=float(cs.min()), vs_flash1024_max_abs=float(np.abs(y - yb).max()),
                                         kv_bytes=kv.nbytes(P_TOK), runs=[])
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
        R["a_last_all"] = [r["chunks"][-1][1] for r in R["runs"]]
        R.pop("runs")
        print(f"{name:16s} total {R['total_med']*1e3:7.1f} ms [{R['total_min']*1e3:.1f}-{R['total_max']*1e3:.1f}] "
              f"attn {R['attn_total_med']*1e3:7.1f} ms [{R['attn_total_min']*1e3:.1f}-{R['attn_total_max']*1e3:.1f}] "
              f"kv {R['kv_bytes']/2**20:.1f} MiB max_abs {R['max_abs']:.2e} rel {R['rel_fro']:.2e} mincos {R['min_cos']:.6f} "
              f"vs_flash1024 {R['vs_flash1024_max_abs']:.1e}")
        print("   ctx  :", " ".join(f"{(i+1)*C:5d}" for i in range(NC)))
        print("   w ms :", " ".join(f"{w*1e3:5.0f}" for w in R["w_med"]))
        print("   a ms :", " ".join(f"{a*1e3:5.0f}" for a in R["a_med"]))
        print("   share:", " ".join(f"{a/(a+w):5.2f}" for w, a in zip(R["w_med"], R["a_med"])))
        print("   last-chunk attn runs ms:", " ".join(f"{a*1e3:.0f}" for a in R["a_last_all"]))
    res["merge_bytes_max"] = STAT.get("merge_bytes", 0)
    print(f"weights u32 {u32_bytes/2**20:.1f} MiB; split merge buffers max {STAT.get('merge_bytes',0)/2**20:.1f} MiB")
    json.dump(res, open(out, "w"), indent=1)
main()
```

</details>

<details><summary>diag.py</summary>

```python
#!/usr/bin/env python3
"""Last-chunk (ctx 8192, C=512) attention-only scaling diag: serial (1 thread) vs W=8 per-head / split8 / qsplit4.
Reuses splitkv kernels without running the prefill. Usage: OPENBLAS_NUM_THREADS=1 python3 diag.py"""
import sys, time, statistics, types
import numpy as np
src = open("splitkv.py").read().replace("\nmain()\n", "\n")
m = types.ModuleType("sk"); exec(compile(src, "splitkv.py", "exec"), m.__dict__)
rng = np.random.default_rng(20261118)
kv = m.KV("a8"); C = m.C; c0 = m.P_TOK - C
for a in range(0, m.P_TOK, C):
    kv.append(rng.standard_normal((C, m.KVH * m.HD)).astype(np.float32), rng.standard_normal((C, m.KVH * m.HD)).astype(np.float32))
q = rng.standard_normal((C, m.D)).astype(np.float32)
qh = q.reshape(C, m.KVH, m.G, m.HD).transpose(1, 2, 0, 3); out = np.empty((C, m.KVH, m.G, m.HD), np.float32)
def serial():
    for h in range(m.KVH):
        Q = qh[h].reshape(m.G * C, m.HD) * m.SC; qpos = np.tile(c0 + np.arange(C), m.G)[:, None]
        mm_, l, acc = m.flash_part(kv, Q, qpos, h, c0, range(8), 1024)
def deq_only():
    for h in range(m.KVH):
        for t in range(8): kv.get(h, t * 1024, (t + 1) * 1024)
F = {"serial_1thread": serial, "dequant_only_1thread": deq_only,
     "flash1024_W8": lambda: m.KERN["flash1024"](kv, qh, c0, out), "split8_1024_W8": lambda: m.KERN["split8_1024"](kv, qh, c0, out),
     "qsplit4_1024_W8": lambda: m.KERN["qsplit4_1024"](kv, qh, c0, out)}
for f in F.values(): f()
R = {k: [] for k in F}
for r in range(9):
    for k, f in F.items():
        t0 = time.perf_counter(); f(); R[k].append(time.perf_counter() - t0)
for k, v in R.items():
    print(f"{k:22s} median {statistics.median(v)*1e3:6.1f} ms [{min(v)*1e3:.1f}-{max(v)*1e3:.1f}]")
s = statistics.median(R["serial_1thread"])
for k in ("flash1024_W8", "split8_1024_W8", "qsplit4_1024_W8"):
    print(f"{k:22s} parallel speedup vs serial {s/statistics.median(R[k]):.2f}x (ideal 5x per-head cap / 8x)")
```

</details>

## Results

Attention = summed over 16 chunks; ratios vs `a8+flash1024` of the same run. Error identical for all paths: max abs 1.37e-2, rel 1.12e-4, min cos 0.999997 vs f32 exact; KV 10.6 MiB for all.

| path | tasks @8192 | run 1 attn ms med [min–max] | run 2 attn ms med [min–max] | attn speed-up r1 / r2 | total prefill ms r1 / r2 | total x r1 / r2 | ctx-8192 chunk attn ms r1 / r2 | share @8192 r1 / r2 | max abs vs flash1024 |
|---|---|---|---|---|---|---|---|---|---|
| flash1024 (02:55) | 5 | 791 [742–839] | 912 [801–958] | 1.00 / 1.00 | 2504 / 2888 | 1.00 / 1.00 | 79 / 90 | 0.45 / 0.45 | 0 |
| split2_1024 | 10 | 759 [744–796] | 804 [783–834] | 1.04 / 1.14 | 2479 / 2706 | 1.01 / 1.07 | 82 / 86 | 0.45 / 0.45 | 8.3e-7 |
| split4_1024 | 20 | 666 [657–671] | 703 [668–814] | 1.19 / 1.30 | 2447 / 2675 | 1.02 / 1.08 | 67 / 68 | 0.40 / 0.39 | 7.2e-7 |
| **split8_1024** | 40 | **627 [594–640]** | **661 [655–708]** | **1.26 / 1.38** | 2312 / 2521 | 1.08 / 1.15 | 63 / 69 | 0.39 / 0.40 | 7.2e-7 |
| split8_512 | 40 | 630 [616–637] | 672 [651–699] | 1.26 / 1.36 | 2354 / 2603 | 1.06 / 1.11 | 66 / 68 | 0.39 / 0.38 | 9.5e-7 |
| qsplit4_1024 | 20 | 636 [622–663] | 652 [632–687] | 1.24 / 1.40 | 2467 / 2574 | 1.02 / 1.12 | 66 / 69 | 0.39 / 0.39 | 0 |

Diag (ctx-8192 chunk, attention only, 9 reps):

```
serial_1thread         median  375.3 ms [359.6-450.6]
dequant_only_1thread   median    6.3 ms [5.9-7.8]
flash1024_W8           median  106.7 ms [79.1-146.9]
split8_1024_W8         median   70.1 ms [61.7-82.9]
qsplit4_1024_W8        median   68.8 ms [63.7-76.7]
flash1024_W8           parallel speedup vs serial 3.52x (ideal 5x per-head cap / 8x)
split8_1024_W8         parallel speedup vs serial 5.35x (ideal 5x per-head cap / 8x)
qsplit4_1024_W8        parallel speedup vs serial 5.45x (ideal 5x per-head cap / 8x)
```

Observations:
- Split-K does recover parallelism: 3.5x → 5.4x of serial on 8 cores. But the ceiling is ~67% of 8 cores (numpy elementwise exp / mask / rescale passes and memory traffic on S tiles), so the best possible gain over per-head was ~1.5x and the prefill run lands at 1.26–1.38x.
- Small contexts get *slower* with split-K (first chunk 11 → 14–15 ms) from task + merge overhead; split pays only past ctx ≈ 3000. qsplit4 has no small-ctx penalty (10–11 ms) because it needs no merge.
- The per-q-head split matches split8 at 8192 despite 4x dequant work: per-tile A8 dequant is ~6 ms of 375 ms serial (1.7%), so redundant dequant is nearly free; the time is in QKᵀ, exp and PV.
- Using the min of each spread (less load-sensitive): run 2 flash1024 801 vs split8 655 ms = 1.22x, consistent with run 1's 1.26x. Run 2's 1.38–1.40x medians are inflated by a noisier baseline.
- Memory: KV unchanged; split8 holds up to 40.6 MiB of fp32 m / l / acc partials per chunk (5 heads × 8 splits × 2048 × 130 × 4 B), ~4x the KV cache itself. qsplit4 adds none.

<details><summary>run1.txt</summary>

```
a8+flash1024     total  2503.9 ms [2419.2-2658.4] attn   791.1 ms [741.6-839.0] kv 10.6 MiB max_abs 1.37e-02 rel 1.12e-04 mincos 0.999997 vs_flash1024 0.0e+00
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :    99    99   103   106   101   112   105   107   108   109   109   111   102   102    98    97
   a ms :    11    18    22    30    32    39    43    45    51    54    58    64    70    74    77    79
   share:  0.10  0.15  0.17  0.22  0.24  0.26  0.29  0.30  0.32  0.33  0.35  0.37  0.40  0.42  0.44  0.45
   last-chunk attn runs ms: 76 78 84 80
a8+split2_1024   total  2478.7 ms [2451.8-2572.4] attn   758.9 ms [744.2-796.4] kv 10.6 MiB max_abs 1.37e-02 rel 1.12e-04 mincos 0.999997 vs_flash1024 8.3e-07
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :   102    98    97   106   103   103   109    99   105    96    95   110   101   110    99    99
   a ms :    14    20    22    27    33    37    40    47    48    55    57    64    67    70    77    82
   share:  0.12  0.17  0.18  0.21  0.25  0.26  0.27  0.32  0.31  0.37  0.38  0.37  0.40  0.39  0.44  0.45
   last-chunk attn runs ms: 80 80 89 83
a8+split4_1024   total  2446.7 ms [2364.3-2484.8] attn   666.2 ms [657.2-671.4] kv 10.6 MiB max_abs 1.37e-02 rel 1.12e-04 mincos 0.999997 vs_flash1024 7.2e-07
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :    97   100    99   101   104   107   106   108   117   103   107   111   105   105   101   101
   a ms :    14    22    21    27    27    30    34    40    46    50    50    53    56    61    62    67
   share:  0.13  0.18  0.18  0.21  0.21  0.22  0.24  0.27  0.28  0.33  0.32  0.33  0.35  0.37  0.38  0.40
   last-chunk attn runs ms: 64 70 69 66
a8+split8_1024   total  2311.7 ms [2195.6-2412.6] attn   627.0 ms [593.9-640.2] kv 10.6 MiB max_abs 1.37e-02 rel 1.12e-04 mincos 0.999997 vs_flash1024 7.2e-07
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :   100   104    99    96    97   101   108    94    99    95    95   104    99   103    98   100
   a ms :    15    20    21    27    27    29    34    37    40    45    45    48    53    57    55    63
   share:  0.13  0.16  0.18  0.22  0.21  0.23  0.24  0.28  0.29  0.32  0.32  0.32  0.35  0.36  0.36  0.39
   last-chunk attn runs ms: 62 64 62 68
a8+split8_512    total  2353.6 ms [2304.1-2559.3] attn   629.5 ms [615.8-636.9] kv 10.6 MiB max_abs 1.37e-02 rel 1.12e-04 mincos 0.999997 vs_flash1024 9.5e-07
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :    98   101   105   107   101   102   100   101   105    99   107   109   102   104   104   103
   a ms :    14    17    20    24    28    31    34    38    41    44    47    50    53    56    59    66
   share:  0.13  0.15  0.16  0.18  0.22  0.23  0.25  0.27  0.28  0.30  0.31  0.32  0.34  0.35  0.36  0.39
   last-chunk attn runs ms: 70 66 65 64
a8+qsplit4_1024  total  2466.6 ms [2304.4-2688.6] attn   636.1 ms [621.5-662.7] kv 10.6 MiB max_abs 1.37e-02 rel 1.12e-04 mincos 0.999997 vs_flash1024 0.0e+00
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :    98   101   103   112   103   108   107   107   131   115   108   125   101   109    99   103
   a ms :    10    15    19    23    27    31    35    39    43    46    48    53    54    59    64    66
   share:  0.10  0.13  0.15  0.17  0.21  0.22  0.25  0.27  0.25  0.29  0.31  0.30  0.35  0.35  0.39  0.39
   last-chunk attn runs ms: 66 66 64 68
weights u32 16.6 MiB; split merge buffers max 40.6 MiB
```

</details>

<details><summary>run2.txt</summary>

```
a8+flash1024     total  2887.6 ms [2578.0-3165.2] attn   912.4 ms [800.7-958.2] kv 10.6 MiB max_abs 1.37e-02 rel 1.12e-04 mincos 0.999997 vs_flash1024 0.0e+00
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :   110   112   110   112   112   108   123   119   114   138   120   125   118   120   120   110
   a ms :    11    19    24    32    37    42    50    52    62    63    74    67    86    89    90    90
   share:  0.09  0.14  0.18  0.22  0.25  0.28  0.29  0.30  0.35  0.31  0.38  0.35  0.42  0.43  0.43  0.45
   last-chunk attn runs ms: 85 101 87 93
a8+split2_1024   total  2706.0 ms [2542.8-2787.6] attn   803.7 ms [782.7-834.2] kv 10.6 MiB max_abs 1.37e-02 rel 1.12e-04 mincos 0.999997 vs_flash1024 8.3e-07
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :   107   111   115   101   115   115   109   107   114   104   105   114   111   115   111   106
   a ms :    14    21    23    28    35    38    41    48    52    56    58    66    70    74    79    86
   share:  0.12  0.16  0.16  0.21  0.23  0.25  0.27  0.31  0.31  0.35  0.36  0.37  0.39  0.39  0.42  0.45
   last-chunk attn runs ms: 85 86 85 91
a8+split4_1024   total  2674.7 ms [2477.6-2849.1] attn   703.2 ms [668.3-814.3] kv 10.6 MiB max_abs 1.37e-02 rel 1.12e-04 mincos 0.999997 vs_flash1024 7.2e-07
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :   112   108   116   125   118   120   118   110   123   103   113   122   118   108   105   107
   a ms :    14    20    23    29    30    36    36    40    47    48    57    59    60    66    65    68
   share:  0.11  0.16  0.16  0.19  0.20  0.23  0.24  0.27  0.28  0.32  0.33  0.33  0.34  0.38  0.38  0.39
   last-chunk attn runs ms: 70 67 66 69
a8+split8_1024   total  2520.6 ms [2439.8-2767.8] attn   660.6 ms [654.7-708.1] kv 10.6 MiB max_abs 1.37e-02 rel 1.12e-04 mincos 0.999997 vs_flash1024 7.2e-07
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :   111   110   106   110   114    99   104   108   116   110   110   129   111   115   109   104
   a ms :    15    21    22    30    28    30    35    41    41    47    51    54    56    64    62    69
   share:  0.12  0.16  0.17  0.21  0.20  0.23  0.25  0.27  0.26  0.30  0.32  0.29  0.34  0.36  0.36  0.40
   last-chunk attn runs ms: 68 67 71 73
a8+split8_512    total  2603.1 ms [2532.5-2697.8] attn   672.4 ms [651.3-698.5] kv 10.6 MiB max_abs 1.37e-02 rel 1.12e-04 mincos 0.999997 vs_flash1024 9.5e-07
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :   102   116   123   125   114   109   118   127   131   111   109   118   103   107   104   113
   a ms :    14    17    22    26    32    33    37    43    46    48    50    52    56    58    63    68
   share:  0.12  0.13  0.15  0.17  0.22  0.23  0.24  0.25  0.26  0.30  0.32  0.31  0.35  0.35  0.38  0.38
   last-chunk attn runs ms: 72 67 64 69
a8+qsplit4_1024  total  2573.7 ms [2511.8-2777.9] attn   651.8 ms [632.4-686.8] kv 10.6 MiB max_abs 1.37e-02 rel 1.12e-04 mincos 0.999997 vs_flash1024 0.0e+00
   ctx  :   512  1024  1536  2048  2560  3072  3584  4096  4608  5120  5632  6144  6656  7168  7680  8192
   w ms :   108   113   119   116   122   108   120   106   116   108   113   119   115   115   106   106
   a ms :    11    15    19    24    27    32    34    39    43    49    50    55    56    62    66    69
   share:  0.10  0.12  0.14  0.17  0.18  0.23  0.22  0.27  0.27  0.31  0.30  0.32  0.33  0.35  0.38  0.39
   last-chunk attn runs ms: 68 70 69 68
weights u32 16.6 MiB; split merge buffers max 40.6 MiB
```

</details>

## Verdict

**not-viable** (speed: 1.19–1.40x on attention, 1.02–1.15x on full prefill, below the 1.5x bar; no memory gain, split-K adds transient bytes). Splitting key tiles across the idle workers with an LSE merge is exact (≤ 9.5e-7 vs per-head, same 0.999997 min cos), and it does cut ctx-8192 attention share from 0.45 to 0.38–0.40, but it misses the ≤ 0.35 target and the 1.5x bar because numpy attention at W = 8 tops out at ~5.4x of serial. If anything is adopted, prefer **qsplit4** (per-q-head tasks): same speed as split8 at long context, no merge, no extra buffers, and no small-context penalty. Lesson: once attention is tiled, more threads give ≤ 1.4x; further wins need fewer passes per tile (fused exp/sum, int8 QKᵀ), not more workers.

## Links

- previous: [[2026-11-18_0255_flash-tiled-online-softmax-attention-per-tile-a8-g64-kv-dequant-vs-whole-context-f16-f32-a8-chunked-prefill-ctx-8192-u32-lut-full-bitnetd-layer]]
- this wave's pair files: none — Quant Chief 05:05 assignment pair_A no gap, pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-18T12:05:00Z); roadmap `JuniorHome docs/ROADMAP.md` blob 4efb666, `JuniorLLM docs/BETA_TO_OS.md` blob bac6923
- bench dir: `/workspace/bench0555_1118` (splitkv.py, diag.py, run1.txt, run2.txt, diag.txt, results_run1.json, results_run2.json)

## Next question

Move from prefill to decode, where flash-decoding is meant to win: for a single-token decode step (B = 1, 4 q rows per KV head) over an 8192-token A8 g64 KV cache, does split-K across the W = 8 pool (LSE merge) make the attention part of the step ≥ 1.5x faster than one task per KV head, and what share of the full u32-LUT layer decode step does attention then take at ctx 2048 / 8192?

close: sha256(prev || this)[:16] = 042ab9f4411ab97f
