---
local: true
date: 2026-11-10
wave: "2026-11-10 18:55 (ran ~18:53–19:05 MT; Chief last_run 2026-11-11T01:08:00Z ~18:08 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, attention, kv-cache, a8, f16, rolling-window, hybrid-cache, numpy, not-viable]
---

# A rolling decoded recent window on top of A8 g=64 KV does not get 1.5x speed at 1.5x less memory than f16: f16 windows are slower than all-A8, and f32 windows only reach 1.26–1.51x when the cache is already bigger than all-f16

## Question

Prior note [[2026-11-10_1455_a8-g64-kv-decode-stage-split-and-fused-int8-score-vs-f32-decoded-512-4096]] found dequant is the largest numpy stage of the A8 g=64 KV step (~40%). Does keeping the most recent W tokens decoded (decode once on append, ring buffer, evict the oldest window token back to A8) and the older T−W tokens as A8 codes recover most of the f32-resident speed at T=2048–4096, while staying at least 1.5x smaller than an all-f16 cache? Viability bar: at least 1.5x faster than all-A8 **and** at least 1.5x smaller than all-f16, at cosine no worse than all-A8. Quant Chief `last_tips.json` `last_run` 2026-11-11T01:08:00Z (~18:08 MT) wrote pair_A/pair_B **no gap**, so this runs the prior note's next question.

## Method

Same bitnetd-shaped attention decode step as the four prior KV notes: B=1, 20 heads × head_dim 128 (hidden 2560), T ∈ {1024, 2048, 4096}, K Student-t(4)×0.5 with 6 outlier channels ×6, V Student-t(4)×0.5, q standard normal, new seed `202611101855`. Paths:

- `all_a8`: A8 g=64 absmax codes with f16 scales for every token, decoded to f32 each step (the 1455 baseline).
- `all_f16_upcast`: whole cache stored as f16, upcast to f32 each step (numpy has no fast f16 matmul).
- `f32_resident`: unquantized f32 K/V, plain matmuls (memory-heavy ceiling).
- `hybrid_W{256,512,1024}_{f16,f32}`: old T−W tokens as A8 codes, recent W tokens stored decoded as f16 (upcast per step) or f32 (used as stored); one joint softmax over both segments.
- `evict_1tok`: per-step window upkeep, re-quantizing the oldest window token to A8 g=64.

Error is cosine and mae vs an f64 reference on unquantized K/V. Timing: 2 warm-ups, 7 repeats × 3 passes, median and min–max; run twice back to back on the shared 8-core box, numpy 2.5.4. A first pair of runs used non-contiguous scale slices for the old segment and showed 30–60 ms outliers at T=4096; the script below makes those slices contiguous and the tables are from the rerun.

Command, from `/workspace/bench1855_1110`: `python3 bench.py results_run1.json` then `python3 bench.py results_run2.json`.

```python
# Hybrid KV cache decode step (B=1, bitnetd shape 20x128): rolling recent window of W tokens kept decoded
# (f16 or f32) + A8 g64 codes for the older T-W tokens, vs all-A8 (f32-decoded), all-f16 (upcast per step), f32-resident.
import numpy as np, time, json, sys
SEED = 202611101855
H, D, G = 20, 128, 64
Gn = D // G
TS = [1024, 2048, 4096]
WS = [256, 512, 1024]
REP, PASS, WARM = 7, 3, 2
rng = np.random.default_rng(SEED)
sc = np.float32(1.0 / np.sqrt(D))
def make_kv(T):
    K = rng.standard_t(4, size=(H, T, D)).astype(np.float32) * 0.5
    ch = rng.choice(D, 6, replace=False); K[:, :, ch] *= 6.0
    V = rng.standard_t(4, size=(H, T, D)).astype(np.float32) * 0.5
    q = rng.standard_normal((H, D)).astype(np.float32)
    return q, K, V
def q8(X):
    Xg = X.reshape(X.shape[:-1] + (Gn, G))
    s = (np.abs(Xg).max(-1, keepdims=True) / 127.0).astype(np.float16); s[s == 0] = 1
    return np.clip(np.rint(Xg / s.astype(np.float32)), -127, 127).astype(np.int8).reshape(X.shape), s[..., 0]
def softmax(x):
    m = x.max(-1, keepdims=True); e = np.exp(x - m); return e / e.sum(-1, keepdims=True)
def a8_score(q, K8, sK):
    T = K8.shape[1]; Kf = K8.astype(np.float32)
    part = np.matmul(Kf.reshape(H, T, Gn, G).transpose(0, 2, 1, 3), q.reshape(H, Gn, G, 1))[..., 0]
    return (part.transpose(0, 2, 1) * sK).sum(-1) * sc
def a8_vfold(p, V8, sV):
    T = p.shape[1]; Vf = V8.astype(np.float32)
    w = (p[:, :, None] * sV).transpose(0, 2, 1)[:, :, None, :]
    return np.matmul(w, Vf.reshape(H, T, Gn, G).transpose(0, 2, 1, 3))[:, :, 0, :].reshape(H, D)
def all_a8(q, K8, sK, V8, sV):
    return a8_vfold(softmax(a8_score(q, K8, sK)), V8, sV)
def all_f16(q, K16, V16):
    p = softmax(np.matmul(K16.astype(np.float32), q[:, :, None])[..., 0] * sc)
    return np.matmul(p[:, None, :], V16.astype(np.float32))[:, 0, :]
def f32_resident(q, K, V):
    p = softmax(np.matmul(K, q[:, :, None])[..., 0] * sc)
    return np.matmul(p[:, None, :], V)[:, 0, :]
def hybrid(q, K8, sK, V8, sV, Kw, Vw):
    # old segment from A8 codes, recent window decoded (f16 upcast per step or f32 as stored); one joint softmax
    Kw32 = Kw if Kw.dtype == np.float32 else Kw.astype(np.float32)
    Vw32 = Vw if Vw.dtype == np.float32 else Vw.astype(np.float32)
    s_old = a8_score(q, K8, sK); s_new = np.matmul(Kw32, q[:, :, None])[..., 0] * sc
    p = softmax(np.concatenate([s_old, s_new], 1)); To = s_old.shape[1]
    return a8_vfold(p[:, :To], V8, sV) + np.matmul(p[:, None, To:], Vw32)[:, 0, :]
def evict(Kw, Vw):  # per-step window upkeep: re-quantize the oldest window token to A8 (ring buffer, no shift)
    return q8(Kw[:, :1, :].astype(np.float32)), q8(Vw[:, :1, :].astype(np.float32))
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
def rec(T, name, f, args, nbytes):
    med, lo, hi = bench(f, *args); o = f(*args)
    r = dict(T=T, path=name, ms=med, ms_min=lo, ms_max=hi, bytes=int(nbytes), cos=cos(o, REF), mae=float(np.mean(np.abs(o - REF))))
    out["rows"].append(r); print(T, name, f"med={med:.3f} [{lo:.3f}-{hi:.3f}] MiB={nbytes/2**20:.2f} cos={r['cos']:.6f}", flush=True)
for T in TS:
    q, K, V = make_kv(T); REF = ref64(q, K, V)
    K8, sK16 = q8(K); V8, sV16 = q8(V); sK = sK16.astype(np.float32); sV = sV16.astype(np.float32)
    a8b = K8.nbytes + V8.nbytes + sK16.nbytes + sV16.nbytes
    rec(T, "all_a8", all_a8, (q, K8, sK, V8, sV), a8b)
    K16, V16 = K.astype(np.float16), V.astype(np.float16)
    rec(T, "all_f16_upcast", all_f16, (q, K16, V16), K16.nbytes + V16.nbytes)
    rec(T, "f32_resident", f32_resident, (q, K, V), K.nbytes + V.nbytes)
    for W in WS:
        if W >= T: continue
        To = T - W
        oK8, osK, oV8, osV = (np.ascontiguousarray(a[:, :To]) for a in (K8, sK, V8, sV))
        ob = oK8.nbytes + oV8.nbytes + 2 * (osK.size + osV.size)
        for wdt, tag in ((np.float16, "f16"), (np.float32, "f32")):
            Kw = np.ascontiguousarray(K[:, To:]).astype(wdt); Vw = np.ascontiguousarray(V[:, To:]).astype(wdt)
            rec(T, f"hybrid_W{W}_{tag}", hybrid, (q, oK8, osK, oV8, osV, Kw, Vw), ob + Kw.nbytes + Vw.nbytes)
            if T == TS[-1] and W == WS[0]:
                med, lo, hi = bench(evict, Kw, Vw); out["rows"].append(dict(T=T, path=f"evict_1tok_{tag}", ms=med, ms_min=lo, ms_max=hi))
                print("evict", tag, med, flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
```

## Results

| T | path | MiB | vs all-f16 bytes | ms run1 / run2 median (run1 min–max) | speed vs all-A8 run1 / run2 | cos vs f64 | mae |
|---|---|---|---|---|---|---|---|
| 1024 | all_a8 | 5.16 | 1.94x | 2.019 / 2.503 (1.989–2.033) | 1.00x / 1.00x | 0.999596 | 1.55e-03 |
| 1024 | all_f16_upcast | 10.00 | 1.00x | 6.153 / 5.761 (6.016–6.220) | 0.33x / 0.43x | 1.000000 | 2.72e-05 |
| 1024 | f32_resident | 20.00 | 0.50x | 0.699 / 0.662 (0.692–0.757) | 2.89x / 3.78x | 1.000000 | 1.56e-08 |
| 1024 | hybrid_W256_f16 | 6.37 | 1.57x | 3.010 / 2.990 (2.924–3.283) | 0.67x / 0.84x | 0.999640 | 1.39e-03 |
| 1024 | hybrid_W256_f32 | 8.87 | 1.13x | 1.771 / 1.676 (1.737–1.908) | 1.14x / 1.49x | 0.999640 | 1.39e-03 |
| 1024 | hybrid_W512_f16 | 7.58 | 1.32x | 4.158 / 4.024 (4.100–4.406) | 0.49x / 0.62x | 0.999692 | 1.24e-03 |
| 1024 | hybrid_W512_f32 | 12.58 | 0.80x | 1.401 / 1.321 (1.362–1.499) | 1.44x / 1.89x | 0.999692 | 1.24e-03 |
| 2048 | all_a8 | 10.31 | 1.94x | 4.264 / 3.987 (4.094–4.850) | 1.00x / 1.00x | 0.999881 | 1.92e-03 |
| 2048 | all_f16_upcast | 20.00 | 1.00x | 12.170 / 11.843 (11.171–12.795) | 0.35x / 0.34x | 1.000000 | 4.42e-05 |
| 2048 | f32_resident | 40.00 | 0.50x | 1.369 / 1.287 (1.330–1.691) | 3.11x / 3.10x | 1.000000 | 2.57e-08 |
| 2048 | hybrid_W256_f16 | 11.52 | 1.74x | 5.029 / 4.846 (4.851–5.618) | 0.85x / 0.82x | 0.999949 | 1.12e-03 |
| 2048 | hybrid_W256_f32 | 14.02 | 1.43x | 3.675 / 3.536 (3.611–3.772) | 1.16x / 1.13x | 0.999950 | 1.13e-03 |
| 2048 | hybrid_W512_f16 | 12.73 | 1.57x | 6.046 / 5.871 (6.004–6.082) | 0.71x / 0.68x | 0.999952 | 1.06e-03 |
| 2048 | hybrid_W512_f32 | 17.73 | 1.13x | 3.679 / 3.222 (3.327–3.974) | 1.16x / 1.24x | 0.999953 | 1.07e-03 |
| 2048 | hybrid_W1024_f16 | 15.16 | 1.32x | 8.815 / 7.994 (8.326–9.484) | 0.48x / 0.50x | 0.999955 | 9.79e-04 |
| 2048 | hybrid_W1024_f32 | 25.16 | 0.80x | 2.822 / 2.691 (2.779–3.116) | 1.51x / 1.48x | 0.999956 | 9.85e-04 |
| 4096 | all_a8 | 20.62 | 1.94x | 9.227 / 8.420 (9.023–9.537) | 1.00x / 1.00x | 0.999813 | 3.79e-03 |
| 4096 | all_f16_upcast | 40.00 | 1.00x | 32.231 / 29.592 (31.028–37.124) | 0.29x / 0.28x | 1.000000 | 6.88e-05 |
| 4096 | f32_resident | 80.00 | 0.50x | 1.472 / 1.317 (1.420–1.561) | 6.27x / 6.39x | 1.000000 | 5.39e-08 |
| 4096 | hybrid_W256_f16 | 21.84 | 1.83x | 11.863 / 11.863 (11.627–12.245) | 0.78x / 0.71x | 0.999814 | 3.74e-03 |
| 4096 | hybrid_W256_f32 | 24.34 | 1.64x | 10.451 / 10.521 (10.405–10.867) | 0.88x / 0.80x | 0.999814 | 3.74e-03 |
| 4096 | hybrid_W512_f16 | 23.05 | 1.74x | 12.387 / 12.366 (12.308–13.237) | 0.74x / 0.68x | 0.999816 | 3.70e-03 |
| 4096 | hybrid_W512_f32 | 28.05 | 1.43x | 9.772 / 9.748 (9.722–10.227) | 0.94x / 0.86x | 0.999816 | 3.70e-03 |
| 4096 | hybrid_W1024_f16 | 25.47 | 1.57x | 12.193 / 11.987 (11.850–13.124) | 0.76x / 0.70x | 0.999845 | 3.11e-03 |
| 4096 | hybrid_W1024_f32 | 35.47 | 1.13x | 6.675 / 6.689 (6.634–8.071) | 1.38x / 1.26x | 0.999845 | 3.10e-03 |

- evict_1tok_f16 (T=4096, W=256): 39 / 36 us per step (run1 / run2)
- evict_1tok_f32 (T=4096, W=256): 34 / 29 us per step (run1 / run2)

Bytes count K and V codes or floats plus f16 group scales. Error columns are identical across runs (same seed). Run2 T=1024 `all_a8` (2.503 ms vs 2.019 ms run1) was hit by box contention, so run2's T=1024 speed ratios are inflated; run1 is the representative row there.

## Verdict

**Not viable.** No hybrid setting clears both bars. An f16 window is always slower than all-A8 (0.48–0.85x), because in numpy upcasting f16 to f32 costs more than upcasting int8 (the all-f16 cache is 0.28–0.43x of all-A8). An f32 window does speed things up once it covers a large share of the cache, but the bytes go with it: the best case, W=1024 f32, is 1.48–1.51x faster than all-A8 at T=2048 while holding 25.16 MiB, which is more than the 20.00 MiB all-f16 cache (0.80x); at T=4096 it is 1.26–1.38x at 35.47 MiB, only 1.13x smaller than f16. The settings that do stay 1.5x+ smaller than f16 (W=256 at T=2048/4096, W=512 at T=4096) are 0.80–1.16x of all-A8, and at T=4096 small windows are slower than plain all-A8 because the split score, concatenate and split V fold add overhead that the small decoded slice cannot pay back. Accuracy is not the issue: the hybrid cosine matches or slightly beats all-A8 (0.99964–0.99996). Window upkeep is negligible (29–39 µs per step to re-quantize the evicted token). Keep the KV cache at all-A8 g=64; in numpy the only real speed lever is f32-resident (3–6x) at 3.88x the bytes, and the decode-cost problem needs a native int8 kernel rather than a cache layout trick. Ternary stays the weight-memory win and is not in play for KV.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-11T01:08:00Z (~18:08 MT) wrote pair_A/pair_B **no gap** ("18:05 11/10: roadmap blobs unchanged; both pairs no gap"; JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). `llc_skip_paths`: JuniorHome `3ec3122` (00:19Z, nico juniorcloudllc) overnight-owned T13+, left to LLC. JuniorLLM tip at push time: [916c926](https://github.com/cloudcover95/JuniorLLM/commit/916c926) (the prior findings note, not touched here).

## Next question

KV layout is settled for numpy and the weight-side kernels are covered by earlier notes, so put the pieces together: for one full bitnetd layer at B=1 (W1.58 2-bit packed attention projections and FFN at hidden 2560 / FFN 6912, plus A8 g=64 KV at T=1024–4096), what share of resident bytes and of step time goes to weights vs KV, and at what context length does KV overtake packed weights as the binding memory cost on an 8 GiB device?

close: 41381f265b462a66 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-10_1455)
