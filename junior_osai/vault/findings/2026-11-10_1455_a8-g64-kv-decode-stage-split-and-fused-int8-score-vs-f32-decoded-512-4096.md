---
local: true
date: 2026-11-10
wave: "2026-11-10 14:55 (ran ~14:57–15:10 MT; Chief last_run 2026-11-10T21:11:00Z ~14:11 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, attention, kv-cache, a8, int8, stage-split, numpy, not-viable]
---

# The A8 g=64 KV step is ~40% dequant and ~35% score; a fused int8 score (int32 accumulate) is 0.64–0.83x of the f32-decoded step in numpy, not 1.5x faster

## Question

Prior note [[2026-11-10_1055_hadamard-rotated-kv-a4-g32-k4v4-trit-v-vs-all-a8-attention-512-4096]] settled the KV cache at A8 g=64 for K and V. Where does the remaining all-A8 decode-step time go (dequant, score matmul, softmax, V fold) at 512–4096 cached tokens, and does a fused int8 score (q quantized to int8 per group, int8 K codes, int32 accumulate, one float multiply per token and group) beat the f32-decoded score path by 1.5x in numpy? Quant Chief `last_tips.json` `last_run` 2026-11-10T21:11:00Z (~14:11 MT) wrote pair_A/pair_B **no gap**, so this runs the prior note's next question.

## Method

Same bitnetd-shaped attention decode step as the three prior notes: B=1, 20 heads × head_dim 128 (hidden 2560), T ∈ {512, 1024, 2048, 4096}, K Student-t(4)×0.5 with 6 outlier channels ×6, V Student-t(4)×0.5, q standard normal, new seed `202611101455`. K and V are A8 g=64 absmax codes with f16 scales. Stages are timed separately on pre-computed inputs; the full f32-decoded step is the prior note's baseline (decode codes to f32, grouped matmul score with scale fold, softmax, scale-folded V matmul). The fused path replaces dequant-K plus f32 score with an int32-accumulated int8 score (tried both `np.matmul` on int32 and `np.einsum` on int16 with `dtype=int32`); V stays f32-decoded. The f32-resident step (unquantized K/V, plain matmuls) is shown as the memory-heavy ceiling. Error is cosine and mae vs an f64 reference on unquantized K/V. Timing: 2 warm-ups, 7 repeats × 3 passes, median and min–max; the script was run twice back to back on the shared 8-core box, numpy 2.5.4.

Command, from `/workspace/bench1455_1110`: `python3 bench.py results_run1.json` then `python3 bench.py results_run2.json`.

```python
# All-A8 g64 KV attention decode step (B=1, bitnetd shape 20x128): stage split (dequant, score, softmax, V fold)
# and fused int8 score (int8 q x int8 K codes, int32 accumulate, per-group scale) vs f32-decoded score path, plus f32-resident ref.
import numpy as np, time, json, sys
SEED = 202611101455
H, D, G = 20, 128, 64
Gn = D // G
TS = [512, 1024, 2048, 4096]
REP, PASS, WARM = 7, 3, 2
rng = np.random.default_rng(SEED)
sc = np.float32(1.0 / np.sqrt(D))
def make_kv(T):
    K = rng.standard_t(4, size=(H, T, D)).astype(np.float32) * 0.5
    ch = rng.choice(D, 6, replace=False); K[:, :, ch] *= 6.0
    V = rng.standard_t(4, size=(H, T, D)).astype(np.float32) * 0.5
    q = rng.standard_normal((H, D)).astype(np.float32)
    return q, K, V
def q8(X):  # X (..., D) -> int8 codes, f16 scales (..., Gn)
    Xg = X.reshape(X.shape[:-1] + (Gn, G))
    s = (np.abs(Xg).max(-1, keepdims=True) / 127.0).astype(np.float16); s[s == 0] = 1
    return np.clip(np.rint(Xg / s.astype(np.float32)), -127, 127).astype(np.int8).reshape(X.shape), s[..., 0]
def softmax(x):
    m = x.max(-1, keepdims=True); e = np.exp(x - m); return e / e.sum(-1, keepdims=True)
# stages of the f32-decoded path (same as 1055 note baseline)
def st_dequant(K8, V8): return K8.astype(np.float32), V8.astype(np.float32)
def st_score(q, Kf, sK):
    T = Kf.shape[1]
    part = np.matmul(Kf.reshape(H, T, Gn, G).transpose(0, 2, 1, 3), q.reshape(H, Gn, G, 1))[..., 0]
    return (part.transpose(0, 2, 1) * sK).sum(-1) * sc
def st_vfold(p, Vf, sV):
    T = p.shape[1]
    w = (p[:, :, None] * sV).transpose(0, 2, 1)[:, :, None, :]
    return np.matmul(w, Vf.reshape(H, T, Gn, G).transpose(0, 2, 1, 3))[:, :, 0, :].reshape(H, D)
def f32_decoded(q, K8, sK, V8, sV):
    Kf, Vf = st_dequant(K8, V8); return st_vfold(softmax(st_score(q, Kf, sK)), Vf, sV)
# fused int8 score: quantize q per group to int8, int32 accumulate per group, one float multiply per (t, group)
def int8_score(q, K8, sK):
    T = K8.shape[1]
    q8c, sq = q8(q)  # (H,D) int8, (H,Gn)
    acc = np.matmul(K8.reshape(H, T, Gn, G).transpose(0, 2, 1, 3).astype(np.int32), q8c.reshape(H, Gn, G, 1).astype(np.int32))[..., 0]
    return (acc.transpose(0, 2, 1).astype(np.float32) * (sK * sq.astype(np.float32)[:, None, :])).sum(-1) * sc
def int8_score_i16(q, K8, sK):  # int16 operands (smaller widen), int32 accumulate via einsum
    T = K8.shape[1]
    q8c, sq = q8(q)
    acc = np.einsum('htgk,hgk->htg', K8.reshape(H, T, Gn, G).astype(np.int16), q8c.reshape(H, Gn, G).astype(np.int16), dtype=np.int32)
    return (acc.astype(np.float32) * (sK * sq.astype(np.float32)[:, None, :])).sum(-1) * sc
def fused_int8(q, K8, sK, V8, sV):
    return st_vfold(softmax(int8_score(q, K8, sK)), V8.astype(np.float32), sV)
def f32_resident(q, K, V):
    p = softmax(np.matmul(K, q[:, :, None])[..., 0] * sc)
    return np.matmul(p[:, None, :], V)[:, 0, :]
def ref64(q, K, V):
    q, K, V = (a.astype(np.float64) for a in (q, K, V))
    p = softmax(np.einsum('hd,htd->ht', q, K) / np.sqrt(D)); return np.einsum('ht,htd->hd', p, V)
def cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
def bench(f, *a):
    for _ in range(WARM): f(*a)
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter()
        for _ in range(PASS): f(*a)
        ts.append((time.perf_counter() - t0) / PASS * 1e3)
    return float(np.median(ts)), float(min(ts)), float(max(ts))
out = {"seed": SEED, "numpy": np.__version__, "rows": []}
for T in TS:
    q, K, V = make_kv(T); ref = ref64(q, K, V)
    K8, sK16 = q8(K); V8, sV16 = q8(V); sK = sK16.astype(np.float32); sV = sV16.astype(np.float32)
    Kf, Vf = st_dequant(K8, V8); s = st_score(q, Kf, sK); p = softmax(s)
    stages = {
        "stage_dequant_KV": (st_dequant, (K8, V8)),
        "stage_score_f32": (st_score, (q, Kf, sK)),
        "stage_softmax": (softmax, (s,)),
        "stage_vfold": (st_vfold, (p, Vf, sV)),
        "score_int8_i32matmul": (int8_score, (q, K8, sK)),
        "score_int8_i16einsum": (int8_score_i16, (q, K8, sK)),
        "step_f32_decoded": (f32_decoded, (q, K8, sK, V8, sV)),
        "step_fused_int8_score": (fused_int8, (q, K8, sK, V8, sV)),
        "step_f32_resident": (f32_resident, (q, K, V)),
    }
    for name, (f, args) in stages.items():
        med, lo, hi = bench(f, *args)
        r = dict(T=T, path=name, ms=med, ms_min=lo, ms_max=hi)
        if name.startswith("step"):
            o = f(*args); r.update(cos=cos(o, ref), mae=float(np.mean(np.abs(o - ref))))
            r["bytes"] = int(K.nbytes + V.nbytes) if name == "step_f32_resident" else int(K8.nbytes + V8.nbytes + sK16.nbytes + sV16.nbytes)
        if name.startswith("score_int8"):
            o = f(*args); r["score_maxabs_err_vs_f32"] = float(np.max(np.abs(o - s)))
        out["rows"].append(r)
        print(T, name, f"med={med:.3f} [{lo:.3f}-{hi:.3f}]", {k: v for k, v in r.items() if k in ('cos', 'mae', 'bytes', 'score_maxabs_err_vs_f32')}, flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
```

## Results

Stage split of the f32-decoded A8 g=64 step (run1 medians, share of stage sum):

| T | dequant K+V | score f32 | softmax | V fold | stage sum | step f32-decoded |
|---|---|---|---|---|---|---|
| 512 | 0.437 ms (42%) | 0.372 ms (36%) | 0.023 ms (2%) | 0.212 ms (20%) | 1.044 ms | 1.097 / 1.014 ms |
| 1024 | 0.882 ms (41%) | 0.761 ms (36%) | 0.039 ms (2%) | 0.448 ms (21%) | 2.131 ms | 2.122 / 2.060 ms |
| 2048 | 1.741 ms (41%) | 1.521 ms (36%) | 0.053 ms (1%) | 0.919 ms (22%) | 4.235 ms | 4.221 / 8.173 ms |
| 4096 | 4.014 ms (39%) | 3.517 ms (35%) | 0.092 ms (1%) | 2.544 ms (25%) | 10.167 ms | 10.170 / 8.691 ms |

| T | path | K+V MiB | ms run1 / run2 median (run1 min–max) | speed vs f32-decoded run1 / run2 | cosine | mae |
|---|---|---|---|---|---|---|
| 512 | f32_decoded | 2.58 | 1.097 / 1.014 (1.056–1.181) | 1.00x / 1.00x | 0.999471 | 2.45e-03 |
| 512 | fused_int8_score | 2.58 | 1.633 / 1.575 (1.585–1.685) | 0.67x / 0.64x | 0.999563 | 2.31e-03 |
| 512 | f32_resident | 10.00 | 0.387 / 0.354 (0.377–0.428) | 2.84x / 2.86x | 1.000000 | 2.08e-08 |
| 1024 | f32_decoded | 5.16 | 2.122 / 2.060 (2.089–2.405) | 1.00x / 1.00x | 0.999936 | 1.55e-03 |
| 1024 | fused_int8_score | 5.16 | 3.140 / 3.081 (3.122–3.212) | 0.68x / 0.67x | 0.999913 | 1.74e-03 |
| 1024 | f32_resident | 20.00 | 0.718 / 0.692 (0.708–0.750) | 2.96x / 2.98x | 1.000000 | 2.58e-08 |
| 2048 | f32_decoded | 10.31 | 4.221 / 8.173 (4.185–5.641) | 1.00x / 1.00x | 0.999481 | 3.53e-03 |
| 2048 | fused_int8_score | 10.31 | 6.148 / 5.971 (5.940–7.100) | 0.69x / 1.37x | 0.999310 | 3.69e-03 |
| 2048 | f32_resident | 40.00 | 1.429 / 1.350 (1.399–1.536) | 2.95x / 6.05x | 1.000000 | 3.57e-08 |
| 4096 | f32_decoded | 20.62 | 10.170 / 8.691 (9.069–12.963) | 1.00x / 1.00x | 0.998870 | 4.40e-03 |
| 4096 | fused_int8_score | 20.62 | 12.185 / 12.173 (11.644–12.700) | 0.83x / 0.71x | 0.998552 | 4.97e-03 |
| 4096 | f32_resident | 80.00 | 1.548 / 1.620 (1.529–1.632) | 6.57x / 5.37x | 1.000000 | 4.72e-08 |

| T | score path | ms run1 / run2 median | vs score f32 (run1) | max abs score error vs f32 |
|---|---|---|---|---|
| 512 | score_f32 (codes pre-decoded) | 0.372 / 0.366 | 1.00x | 0 |
| 512 | score_int8_i32matmul | 1.111 / 1.121 | 0.34x | 0.052 |
| 512 | score_int8_i16einsum | 1.067 / 1.055 | 0.35x | 0.052 |
| 1024 | score_f32 (codes pre-decoded) | 0.761 / 0.727 | 1.00x | 0 |
| 1024 | score_int8_i32matmul | 2.215 / 2.154 | 0.34x | 0.078 |
| 1024 | score_int8_i16einsum | 2.159 / 2.100 | 0.35x | 0.078 |
| 2048 | score_f32 (codes pre-decoded) | 1.521 / 1.515 | 1.00x | 0 |
| 2048 | score_int8_i32matmul | 4.227 / 4.307 | 0.36x | 0.081 |
| 2048 | score_int8_i16einsum | 4.261 / 4.508 | 0.36x | 0.081 |
| 4096 | score_f32 (codes pre-decoded) | 3.517 / 3.215 | 1.00x | 0 |
| 4096 | score_int8_i32matmul | 8.447 / 8.466 | 0.42x | 0.096 |
| 4096 | score_int8_i16einsum | 8.319 / 8.356 | 0.42x | 0.096 |

Error columns are identical across runs (same seed). Run2 T=2048 f32-decoded median (8.173 ms, min–max 5.264–14.587) was hit by box contention; run1 and the stage sum (4.2 ms) are the representative values, so the 1.37x in that row is noise, not a win.

## Verdict

**Not viable.** In numpy the A8 step is dominated by turning codes into floats (dequant ~39–42%) and the score contraction (~35–36%); softmax is 1–2% and the V fold 20–25%. A fused int8 score does not help: numpy has no BLAS path for integer matmul, so the int32-accumulated score runs at 0.34–0.42x of the f32 score on pre-decoded codes, and even against dequant-K plus f32 score (about 0.59 ms at T=512) it is roughly half the speed. The whole fused step is 0.64–0.83x of the f32-decoded step at the same 2.58–20.6 MiB (1.94x less than f16, 3.88x less than f32) with matching cosine (0.9986–0.9999 vs 0.9989–0.9999). The f32-resident step is 2.8–6.6x faster but holds 3.88x more bytes, so in numpy the quantized KV buys memory and pays in time; a native int8 dot kernel (VNNI/SDOT) is what would flip the speed column, which this lane cannot measure without installs. Ternary is not in play for KV at all (prior notes); keep A8 g=64 KV and keep ternary as the weight-memory win.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-10T21:11:00Z (~14:11 MT) wrote pair_A/pair_B **no gap** (roadmap blobs unchanged, JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). `llc_skip_paths`: JuniorHome `3e1f348` (20:11Z, nico juniorcloudllc) re-advances landed UE5, left to LLC. JuniorLLM tip at push time: [ddc852a](https://github.com/cloudcover95/JuniorLLM/commit/ddc852a) (the prior findings note, not touched here).

## Next question

Since dequant is the largest numpy stage, does caching decoded K/V as f16 per block (decode once on append, keep a rolling f16 window of the last W tokens and A8 for the rest) recover most of the f32-resident speed at T=2048–4096 while staying at least 1.5x smaller than an all-f16 cache?

close: b6848f24dc7cca70 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-10_1055)
