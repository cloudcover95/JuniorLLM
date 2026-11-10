---
local: true
date: 2026-11-10
wave: "2026-11-10 02:55 (ran ~02:55–03:05 MT; Chief last_run 2026-11-10T09:13:30Z ~02:13 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, attention, kv-cache, a8, g64, f16, trit, numpy, not-viable]
---

# A8 g=64 KV cache holds cosine ≥ 0.999 at 1.94x less memory than f16, but ternary KV breaks attention (cosine 0.06–0.36)

## Question

Prior note [[2026-11-09_1855_activation-scaling-per-group-g64-g128-winsor-mseclip-a4-a8-bitslice-and-expand-vs-f32]] settled W1.58A8 with g=64 activation scales as the bitnetd contract and asked what that costs in the KV path: does storing the attention K/V cache as A8 g=64 (int8 + one f16 scale per 64) versus f16 hold cosine ≥ 0.999 on a softmax attention step over 512–4096 cached tokens, and what are the memory and B=1 step-time ratios? This run also asks whether going one step further, a ternary g=64 KV cache (2-bit packed trits + one f16 absmean scale per 64), is a real win. Quant Chief `last_tips.json` `last_run` 2026-11-10T09:13:30Z (~02:13 MT) wrote pair_A/pair_B **no gap**, so this runs the prior note's next question.

## Method

One bitnetd-shaped attention decode step at B=1: 20 heads × head_dim 128 (hidden 2560), cached length T ∈ {512, 1024, 2048, 4096}, seed `202611100255`. K is Student-t(4)×0.5 with 6 outlier channels ×6 (keys in real models carry channel outliers), V is Student-t(4)×0.5, q is standard normal. Step: scores = q·Kᵀ/√128, softmax, output = p·V. Error is cosine and mae of the attention output against an f64 reference on the unquantized K/V; it is deterministic.

Storage paths: f32-resident; f16 (cast to f32 for the batched matmul, the way numpy has to run it); A8 g=64 with two kernels, full dequant then batched matmul (`a8g64_deq`) and scale-folded (`a8g64_fold`: per-group partial dots of int8 K with q scaled by the K group scale, and the softmax weights multiplied by the V group scale before the int8 V contraction, so no full dequantized copy is written); ternary g=64 via a 256×4 byte LUT decode then batched matmul. Bytes are the stored K+V arrays including scales. The fold kernel matches the dequant kernel to < 1e-4 relative (asserted). Timing: 2 warm-ups, 7 repeats × 3 passes, median and min–max; run twice back to back on the shared 8-core box, numpy 2.5.4.

Command, from `/workspace/bench0255_1110`: `python3 bench.py results_run1.json` then `python3 bench.py results_run2.json`.

```python
# KV cache storage for bitnetd attention step (B=1 decode): f32 / f16 / A8 g64 (int8 + f16 scale per 64) / trit g64 (2-bit packed + f16 scale per 64)
import numpy as np, time, json, sys
SEED = 202611100255
H, D, G = 20, 128, 64          # 2560 hidden = 20 heads x 128; group 64 along head_dim
TS = [512, 1024, 2048, 4096]
REP, PASS, WARM = 7, 3, 2
rng = np.random.default_rng(SEED)
LUT = np.array([[((b >> (2*i)) & 3) for i in range(4)] for b in range(256)], np.int8) - 1  # code 0,1,2 -> -1,0,+1

def make_kv(T):
    # K with a few outlier channels (as in real attention keys), V Student-t
    K = rng.standard_t(4, size=(H, T, D)).astype(np.float32) * 0.5
    ch = rng.choice(D, 6, replace=False); K[:, :, ch] *= 6.0
    V = rng.standard_t(4, size=(H, T, D)).astype(np.float32) * 0.5
    q = rng.standard_normal((H, D)).astype(np.float32)
    return q, K, V

def q8(X):
    Xg = X.reshape(H, -1, D // G, G)
    s = np.abs(Xg).max(-1, keepdims=True) / 127.0; s[s == 0] = 1
    s16 = s.astype(np.float16)
    c = np.clip(np.rint(Xg / s16.astype(np.float32)), -127, 127).astype(np.int8)
    return c.reshape(X.shape), s16[..., 0]

def qt(X):
    Xg = X.reshape(H, -1, D // G, G)
    s = np.abs(Xg).mean(-1, keepdims=True); s[s == 0] = 1
    s16 = s.astype(np.float16)
    t = np.clip(np.rint(Xg / s16.astype(np.float32)), -1, 1).astype(np.int8) + 1  # 0..2
    t = t.reshape(H, X.shape[1], D // 4, 4).astype(np.uint8)
    packed = t[..., 0] | (t[..., 1] << 2) | (t[..., 2] << 4) | (t[..., 3] << 6)
    return packed, s16[..., 0]

def softmax(x):
    m = x.max(-1, keepdims=True); e = np.exp(x - m); return e / e.sum(-1, keepdims=True)

sc = np.float32(1.0 / np.sqrt(D))
def attn_f32(q, K, V):
    p = softmax(np.einsum('hd,htd->ht', q, K) * sc)
    return np.einsum('ht,htd->hd', p, V)
def attn_f32_bmm(q, K, V):
    s = np.matmul(K, q[:, :, None])[..., 0] * sc
    p = softmax(s)
    return np.matmul(p[:, None, :], V)[:, 0, :]
def attn_f16(q, K16, V16):
    return attn_f32_bmm(q, K16.astype(np.float32), V16.astype(np.float32))
def attn_a8_deq(q, K8, sK, V8, sV):
    T = K8.shape[1]
    Kf = (K8.reshape(H, T, D // G, G).astype(np.float32) * sK[..., None].astype(np.float32)).reshape(H, T, D)
    Vf = (V8.reshape(H, T, D // G, G).astype(np.float32) * sV[..., None].astype(np.float32)).reshape(H, T, D)
    return attn_f32_bmm(q, Kf, Vf)
def attn_a8_fold(q, K8, sK, V8, sV):
    # fold scales: scores = sum_g sK[t,g] * (K8[t,g,:] . q_g); out_g = (p*sV[:,g]) @ V8[:,g,:]
    T = K8.shape[1]
    Kf = K8.astype(np.float32).reshape(H, T, D // G, G)
    qg = q.reshape(H, D // G, G)
    part = np.einsum('htgd,hgd->htg', Kf, qg)
    s = (part * sK.astype(np.float32)).sum(-1) * sc
    p = softmax(s)
    Vf = V8.astype(np.float32).reshape(H, T, D // G, G)
    w = p[:, :, None] * sV.astype(np.float32)
    return np.einsum('htg,htgd->hgd', w, Vf).reshape(H, D)
def attn_trit(q, Kp, sK, Vp, sV):
    T = Kp.shape[1]
    Kt = LUT[Kp].reshape(H, T, D // G, G).astype(np.float32) * sK[..., None].astype(np.float32)
    Vt = LUT[Vp].reshape(H, T, D // G, G).astype(np.float32) * sV[..., None].astype(np.float32)
    return attn_f32_bmm(q, Kt.reshape(H, T, D), Vt.reshape(H, T, D))

def ref64(q, K, V):
    q, K, V = (a.astype(np.float64) for a in (q, K, V))
    s = np.einsum('hd,htd->ht', q, K) / np.sqrt(D); p = softmax(s)
    return np.einsum('ht,htd->hd', p, V), s

def cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))

def bench(f, *a):
    for _ in range(WARM): f(*a)
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter()
        for _ in range(PASS): f(*a)
        ts.append((time.perf_counter() - t0) / PASS * 1e3)
    return float(np.median(ts)), float(min(ts)), float(max(ts))

out = {"seed": SEED, "numpy": np.__version__, "H": H, "D": D, "G": G, "rows": []}
for T in TS:
    q, K, V = make_kv(T)
    ref, sref = ref64(q, K, V)
    K16, V16 = K.astype(np.float16), V.astype(np.float16)
    K8, sK8 = q8(K); V8, sV8 = q8(V)
    Kp, sKt = qt(K); Vp, sVt = qt(V)
    # exactness of fold vs dequant
    a = attn_a8_deq(q, K8, sK8, V8, sV8); b = attn_a8_fold(q, K8, sK8, V8, sV8)
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 1e-4
    nb = lambda *xs: int(sum(x.nbytes for x in xs))
    paths = {
        "f32": (attn_f32_bmm, (q, K, V), nb(K, V)),
        "f16": (attn_f16, (q, K16, V16), nb(K16, V16)),
        "a8g64_deq": (attn_a8_deq, (q, K8, sK8, V8, sV8), nb(K8, sK8, V8, sV8)),
        "a8g64_fold": (attn_a8_fold, (q, K8, sK8, V8, sV8), nb(K8, sK8, V8, sV8)),
        "trit_g64": (attn_trit, (q, Kp, sKt, Vp, sVt), nb(Kp, sKt, Vp, sVt)),
    }
    for name, (f, args, by) in paths.items():
        o = f(*args)
        med, lo, hi = bench(f, *args)
        out["rows"].append(dict(T=T, path=name, bytes=by, ms=med, ms_min=lo, ms_max=hi,
                                cos=cos(o, ref), mae=float(np.mean(np.abs(o - ref)))))
        r = out["rows"][-1]
        print(f"T={T:5d} {name:11s} bytes={by:>10d} med={med:8.3f}ms [{lo:.3f}-{hi:.3f}] cos={r['cos']:.6f} mae={r['mae']:.2e}", flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
```

## Results

Memory ratio and speed ratio are relative to f16 (speed shown run1 / run2); "vs f32" is run2 speed relative to the f32-resident path.

| T | path | K+V MiB | memory vs f16 | median ms run1 / run2 (min–max run2) | speed vs f16 | speed vs f32 | cosine | mae |
|---|---|---|---|---|---|---|---|---|
| 512 | f32 | 10.00 | 0.50x | 0.367 / 0.426 (0.412–0.448) | 8.89x / 7.39x | 1.00x | 1.000000 | 1.60e-08 |
| 512 | f16 | 5.00 | 1.00x | 3.258 / 3.149 (2.945–5.031) | 1.00x / 1.00x | 0.14x | 1.000000 | 3.39e-05 |
| 512 | a8g64_deq | 2.58 | 1.94x | 2.011 / 2.090 (2.045–2.129) | 1.62x / 1.51x | 0.20x | 0.999439 | 1.74e-03 |
| 512 | a8g64_fold | 2.58 | 1.94x | 1.661 / 1.652 (1.611–1.727) | 1.96x / 1.91x | 0.26x | 0.999439 | 1.74e-03 |
| 512 | trit_g64 | 0.70 | 7.11x | 8.678 / 8.278 (8.216–8.415) | 0.38x / 0.38x | 0.05x | 0.363297 | 5.47e-02 |
| 1024 | f32 | 20.00 | 0.50x | 0.695 / 0.760 (0.738–0.781) | 9.15x / 8.80x | 1.00x | 1.000000 | 2.03e-08 |
| 1024 | f16 | 10.00 | 1.00x | 6.360 / 6.689 (6.464–6.792) | 1.00x / 1.00x | 0.11x | 1.000000 | 3.85e-05 |
| 1024 | a8g64_deq | 5.16 | 1.94x | 4.159 / 4.478 (4.360–6.618) | 1.53x / 1.49x | 0.17x | 0.999133 | 2.20e-03 |
| 1024 | a8g64_fold | 5.16 | 1.94x | 3.180 / 3.509 (3.477–3.558) | 2.00x / 1.91x | 0.22x | 0.999133 | 2.20e-03 |
| 1024 | trit_g64 | 1.41 | 7.11x | 16.830 / 17.847 (16.415–18.331) | 0.38x / 0.37x | 0.04x | 0.260874 | 5.36e-02 |
| 2048 | f32 | 40.00 | 0.50x | 1.391 / 1.497 (1.441–1.556) | 8.96x / 8.21x | 1.00x | 1.000000 | 4.58e-08 |
| 2048 | f16 | 20.00 | 1.00x | 12.466 / 12.285 (11.922–14.002) | 1.00x / 1.00x | 0.12x | 1.000000 | 8.35e-05 |
| 2048 | a8g64_deq | 10.31 | 1.94x | 9.818 / 10.240 (9.885–10.690) | 1.27x / 1.20x | 0.15x | 0.999725 | 4.12e-03 |
| 2048 | a8g64_fold | 10.31 | 1.94x | 6.649 / 6.847 (6.524–8.049) | 1.87x / 1.79x | 0.22x | 0.999725 | 4.12e-03 |
| 2048 | trit_g64 | 2.81 | 7.11x | 35.413 / 36.309 (35.640–38.277) | 0.35x / 0.34x | 0.04x | 0.056266 | 1.66e-01 |
| 4096 | f32 | 80.00 | 0.50x | 1.465 / 1.428 (1.427–1.798) | 20.92x / 22.21x | 1.00x | 1.000000 | 3.66e-08 |
| 4096 | f16 | 40.00 | 1.00x | 30.657 / 31.718 (31.125–33.989) | 1.00x / 1.00x | 0.05x | 1.000000 | 6.94e-05 |
| 4096 | a8g64_deq | 20.62 | 1.94x | 23.958 / 27.166 (25.339–28.888) | 1.28x / 1.17x | 0.05x | 0.999405 | 3.97e-03 |
| 4096 | a8g64_fold | 20.62 | 1.94x | 16.133 / 17.178 (16.664–17.896) | 1.90x / 1.85x | 0.08x | 0.999405 | 3.97e-03 |
| 4096 | trit_g64 | 5.62 | 7.11x | 81.046 / 85.314 (81.599–90.959) | 0.38x / 0.37x | 0.02x | 0.064905 | 8.66e-02 |

## Verdict

**Not viable for ternary.** A ternary g=64 KV cache is 7.11x smaller than f16 but wrecks the attention output (cosine 0.36 at T=512 falling to 0.06 at T=4096), because the score error from trit-rounded keys gets amplified by softmax and compounds with length; it is also the slowest path (2.6–3.0x slower than f16 in numpy). The KV cache is not a place for ternary.

The A8 g=64 contract does carry over to the KV cache: cosine 0.99913–0.99973 across 512–4096 tokens (all ≥ 0.999), with exactly 1.94x less K+V memory than f16 (1.03125 vs 2 bytes per element), and the scale-folded kernel runs 1.8–2.0x faster than the f16 path in numpy. That meets the 1.5x memory bar, but it is an int8 result, not a ternary one, and halving the KV cache with int8 is well established; it is still 3.9–12x slower than keeping K/V f32-resident in numpy (run2: 0.26x at T=512, 0.08x at T=4096), so the speed edge over f16 only reflects numpy's slow f16-to-f32 cast. For bitnetd: ternary weights, A8 g=64 activations, A8 g=64 KV. Bounded to numpy on synthetic K/V; a native int8 dot or f16 SIMD kernel would change the speed picture.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-10T09:13:30Z (~02:13 MT) wrote pair_A/pair_B **no gap** (roadmap blobs unchanged, JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). `llc_skip_paths`: JuniorHome `5c8319d` (08:17Z), left to LLC. JuniorLLM tip at push time: [ea840c8](https://github.com/cloudcover95/JuniorLLM/commit/ea840c8) (`rails/closechain/close_chain_verify.py`, not touched here).

## Next question

KV memory per token now dominates at long context: with A8 g=64 KV fixed, does a ternary or 4-bit **value-only** cache (keep K at A8 g=64, quantize only V, where errors are averaged by softmax instead of amplified) hold cosine ≥ 0.999 at 512–4096 tokens, and what memory ratio does it give versus all-A8?

close: b1de58cb6cff721a (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-09_1855)
