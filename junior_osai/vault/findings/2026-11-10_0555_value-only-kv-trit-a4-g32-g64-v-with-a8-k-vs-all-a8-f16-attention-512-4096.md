---
local: true
date: 2026-11-10
wave: "2026-11-10 05:55 (ran ~05:55–06:05 MT; Chief last_run 2026-11-10T12:12:30Z ~05:12 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, attention, kv-cache, value-only, a4, a8, trit, numpy, not-viable]
---

# Value-only ternary or 4-bit V cache does not hold cosine 0.999: trit V cos 0.82 at 1.57x less memory than all-A8, A4 V cos 0.986–0.992 at 1.29–1.32x

## Question

Prior note [[2026-11-10_0255_kv-cache-a8-g64-and-trit-g64-vs-f16-f32-attention-step-512-4096]] fixed the KV contract at A8 g=64 for both K and V and asked: with K kept at A8 g=64, does quantizing only V to ternary (2-bit packed trits + f16 absmean scale per group) or to 4-bit (packed nibbles + f16 absmax scale per group), where the error is mixed by the softmax weights instead of amplified through the scores, hold cosine ≥ 0.999 at 512–4096 cached tokens, and what memory ratio does it give versus all-A8? Quant Chief `last_tips.json` `last_run` 2026-11-10T12:12:30Z (~05:12 MT) wrote pair_A/pair_B **no gap**, so this runs the prior note's next question.

## Method

Same bitnetd-shaped attention decode step as the prior note: B=1, 20 heads × head_dim 128 (hidden 2560), cached length T ∈ {512, 1024, 2048, 4096}, K Student-t(4)×0.5 with 6 outlier channels ×6, V Student-t(4)×0.5, q standard normal, seed `202611100555`. Error is cosine and mae of the attention output against an f64 reference on unquantized K/V (deterministic). Every quantized path uses the same A8 g=64 K and scale-folded scores; V variants are A8 g=64 (baseline all-A8), A4 g=64 and g=32 (symmetric ±7, 16-entry nibble LUT), and trit g=64 and g=32 (absmean scale, 256×4 byte LUT). The V contraction folds the group scale into the softmax weights. f16 K/V is included as the reference store. Bytes are stored K+V arrays including scales. Timing: 2 warm-ups, 7 repeats × 3 passes, median and min–max; the whole script was run twice back to back on the shared 8-core box, numpy 2.5.4.

Command, from `/workspace/bench0555_1110`: `python3 bench.py results_run1.json` then `python3 bench.py results_run2.json`.

```python
# Value-only KV quant for bitnetd attention step (B=1 decode): K fixed at A8 g64; V as A8 g64 / A4 g64 / A4 g32 / trit g64 / trit g32
import numpy as np, time, json, sys
SEED = 202611100555
H, D = 20, 128
TS = [512, 1024, 2048, 4096]
REP, PASS, WARM = 7, 3, 2
rng = np.random.default_rng(SEED)
TLUT = (np.array([[((b >> (2*i)) & 3) for i in range(4)] for b in range(256)], np.int8) - 1).astype(np.float32)
NLUT = (np.array([[((b >> (4*i)) & 15) for i in range(2)] for b in range(256)], np.int16) - 8).astype(np.float32)

def make_kv(T):
    K = rng.standard_t(4, size=(H, T, D)).astype(np.float32) * 0.5
    ch = rng.choice(D, 6, replace=False); K[:, :, ch] *= 6.0
    V = rng.standard_t(4, size=(H, T, D)).astype(np.float32) * 0.5
    q = rng.standard_normal((H, D)).astype(np.float32)
    return q, K, V

def q8(X, G=64):
    Xg = X.reshape(H, -1, D // G, G)
    s = (np.abs(Xg).max(-1, keepdims=True) / 127.0).astype(np.float16); s[s == 0] = 1
    c = np.clip(np.rint(Xg / s.astype(np.float32)), -127, 127).astype(np.int8)
    return c.reshape(X.shape), s[..., 0]
def q4(X, G):
    Xg = X.reshape(H, -1, D // G, G)
    s = (np.abs(Xg).max(-1, keepdims=True) / 7.0).astype(np.float16); s[s == 0] = 1
    c = (np.clip(np.rint(Xg / s.astype(np.float32)), -7, 7).astype(np.int16) + 8).astype(np.uint8).reshape(H, X.shape[1], D // 2, 2)
    return (c[..., 0] | (c[..., 1] << 4)), s[..., 0]
def qt(X, G):
    Xg = X.reshape(H, -1, D // G, G)
    s = np.abs(Xg).mean(-1, keepdims=True).astype(np.float16); s[s == 0] = 1
    t = (np.clip(np.rint(Xg / s.astype(np.float32)), -1, 1).astype(np.int8) + 1).astype(np.uint8).reshape(H, X.shape[1], D // 4, 4)
    return (t[..., 0] | (t[..., 1] << 2) | (t[..., 2] << 4) | (t[..., 3] << 6)), s[..., 0]

def softmax(x):
    m = x.max(-1, keepdims=True); e = np.exp(x - m); return e / e.sum(-1, keepdims=True)
sc = np.float32(1.0 / np.sqrt(D))
def scores_a8(q, K8, sK):
    T = K8.shape[1]; G = D // sK.shape[-1]
    part = np.einsum('htgd,hgd->htg', K8.astype(np.float32).reshape(H, T, D // G, G), q.reshape(H, D // G, G))
    return softmax((part * sK.astype(np.float32)).sum(-1) * sc)
def vfold(p, Vc, sV, G):
    T = p.shape[1]
    w = p[:, :, None] * sV.astype(np.float32)
    return np.einsum('htg,htgd->hgd', w, Vc.reshape(H, T, D // G, G)).reshape(H, D)
def step_a8(q, K8, sK, V8, sV): return vfold(scores_a8(q, K8, sK), V8.astype(np.float32), sV, 64)
def step_a4(q, K8, sK, Vp, sV): G = D // sV.shape[-1]; return vfold(scores_a8(q, K8, sK), NLUT[Vp], sV, G)
def step_t(q, K8, sK, Vp, sV):  G = D // sV.shape[-1]; return vfold(scores_a8(q, K8, sK), TLUT[Vp], sV, G)

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

out = {"seed": SEED, "numpy": np.__version__, "H": H, "D": D, "rows": []}
for T in TS:
    q, K, V = make_kv(T); ref = ref64(q, K, V)
    K8, sK = q8(K)
    V8, sV8 = q8(V); V4a, s4a = q4(V, 64); V4b, s4b = q4(V, 32); Vta, sta = qt(V, 64); Vtb, stb = qt(V, 32)
    K16, V16 = K.astype(np.float16), V.astype(np.float16)
    nb = lambda *xs: int(sum(x.nbytes for x in xs))
    paths = {
        "f16_KV": (lambda q, K16, V16: (lambda p: np.matmul(p[:, None, :], V16.astype(np.float32))[:, 0, :])(softmax(np.matmul(K16.astype(np.float32), q[:, :, None])[..., 0] * sc)), (q, K16, V16), nb(K16, V16)),
        "K8_V8g64": (step_a8, (q, K8, sK, V8, sV8), nb(K8, sK, V8, sV8)),
        "K8_V4g64": (step_a4, (q, K8, sK, V4a, s4a), nb(K8, sK, V4a, s4a)),
        "K8_V4g32": (step_a4, (q, K8, sK, V4b, s4b), nb(K8, sK, V4b, s4b)),
        "K8_Vtg64": (step_t, (q, K8, sK, Vta, sta), nb(K8, sK, Vta, sta)),
        "K8_Vtg32": (step_t, (q, K8, sK, Vtb, stb), nb(K8, sK, Vtb, stb)),
    }
    for name, (f, args, by) in paths.items():
        o = f(*args); med, lo, hi = bench(f, *args)
        r = dict(T=T, path=name, bytes=by, ms=med, ms_min=lo, ms_max=hi, cos=cos(o, ref), mae=float(np.mean(np.abs(o - ref))))
        out["rows"].append(r)
        print(f"T={T:5d} {name:9s} bytes={by:>9d} med={med:8.3f}ms [{lo:.3f}-{hi:.3f}] cos={r['cos']:.6f} mae={r['mae']:.2e}", flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
```

## Results

| T | path | K+V MiB | mem vs f16 | mem vs all-A8 | ms run1 / run2 median (run1 min–max) | speed vs all-A8 run1 / run2 | cosine | mae |
|---|---|---|---|---|---|---|---|---|
| 512 | f16_KV | 5.00 | 1.00x | 0.52x | 3.147 / 3.085 (3.097–4.135) | 0.56x / 0.50x | 1.000000 | 2.57e-05 |
| 512 | K8_V8g64 | 2.58 | 1.94x | 1.00x | 1.766 / 1.553 (1.645–1.987) | 1.00x / 1.00x | 0.999723 | 1.18e-03 |
| 512 | K8_V4g64 | 1.95 | 2.56x | 1.32x | 7.535 / 7.269 (7.480–8.907) | 0.23x / 0.21x | 0.987458 | 8.53e-03 |
| 512 | K8_V4g32 | 1.99 | 2.51x | 1.29x | 7.722 / 7.464 (7.664–7.877) | 0.23x / 0.21x | 0.991555 | 6.99e-03 |
| 512 | K8_Vtg64 | 1.64 | 3.05x | 1.57x | 4.551 / 4.305 (4.472–4.720) | 0.39x / 0.36x | 0.823209 | 3.11e-02 |
| 512 | K8_Vtg32 | 1.68 | 2.98x | 1.53x | 4.658 / 4.337 (4.430–4.741) | 0.38x / 0.36x | 0.829722 | 3.08e-02 |
| 1024 | f16_KV | 10.00 | 1.00x | 0.52x | 6.215 / 5.812 (6.158–6.303) | 0.51x / 0.54x | 1.000000 | 3.56e-05 |
| 1024 | K8_V8g64 | 5.16 | 1.94x | 1.00x | 3.186 / 3.116 (3.132–3.218) | 1.00x / 1.00x | 0.999722 | 1.42e-03 |
| 1024 | K8_V4g64 | 3.91 | 2.56x | 1.32x | 15.171 / 15.725 (14.907–15.472) | 0.21x / 0.20x | 0.989059 | 9.05e-03 |
| 1024 | K8_V4g32 | 3.98 | 2.51x | 1.29x | 15.314 / 15.707 (15.090–15.764) | 0.21x / 0.20x | 0.991981 | 7.70e-03 |
| 1024 | K8_Vtg64 | 3.28 | 3.05x | 1.57x | 9.309 / 9.035 (9.129–9.419) | 0.34x / 0.34x | 0.819602 | 3.31e-02 |
| 1024 | K8_Vtg32 | 3.36 | 2.98x | 1.53x | 9.229 / 9.154 (9.050–9.560) | 0.35x / 0.34x | 0.824556 | 3.29e-02 |
| 2048 | f16_KV | 20.00 | 1.00x | 0.52x | 12.758 / 11.768 (12.485–13.274) | 0.53x / 0.54x | 1.000000 | 3.81e-05 |
| 2048 | K8_V8g64 | 10.31 | 1.94x | 1.00x | 6.722 / 6.365 (6.434–6.831) | 1.00x / 1.00x | 0.999338 | 2.38e-03 |
| 2048 | K8_V4g64 | 7.81 | 2.56x | 1.32x | 31.212 / 29.339 (30.527–34.272) | 0.22x / 0.22x | 0.987299 | 1.08e-02 |
| 2048 | K8_V4g32 | 7.97 | 2.51x | 1.29x | 34.991 / 30.003 (32.838–38.516) | 0.19x / 0.21x | 0.991971 | 8.81e-03 |
| 2048 | K8_Vtg64 | 6.56 | 3.05x | 1.57x | 18.947 / 17.751 (18.235–20.371) | 0.35x / 0.36x | 0.840335 | 3.28e-02 |
| 2048 | K8_Vtg32 | 6.72 | 2.98x | 1.53x | 19.411 / 18.494 (18.522–20.406) | 0.35x / 0.34x | 0.840214 | 3.28e-02 |
| 4096 | f16_KV | 40.00 | 1.00x | 0.52x | 32.828 / 32.961 (31.875–35.052) | 0.46x / 0.50x | 1.000000 | 3.29e-05 |
| 4096 | K8_V8g64 | 20.62 | 1.94x | 1.00x | 15.227 / 16.593 (14.889–16.394) | 1.00x / 1.00x | 0.999866 | 2.47e-03 |
| 4096 | K8_V4g64 | 15.62 | 2.56x | 1.32x | 67.193 / 75.930 (63.720–72.114) | 0.23x / 0.22x | 0.986129 | 2.10e-02 |
| 4096 | K8_V4g32 | 15.94 | 2.51x | 1.29x | 73.309 / 72.581 (69.559–77.046) | 0.21x / 0.23x | 0.990297 | 1.69e-02 |
| 4096 | K8_Vtg64 | 13.12 | 3.05x | 1.57x | 39.673 / 42.588 (39.263–44.218) | 0.38x / 0.39x | 0.817475 | 5.57e-02 |
| 4096 | K8_Vtg32 | 13.44 | 2.98x | 1.53x | 42.433 / 54.265 (41.525–47.573) | 0.36x / 0.31x | 0.819695 | 5.60e-02 |

Error columns are identical across runs (same seed).

## Verdict

**Not viable.** Value-only quantization does not hold the 0.999 bar. Ternary V reaches the 1.5x memory bar over all-A8 (1.57x at g=64, 1.53x at g=32; 3.05x vs f16) but the attention output cosine sits at 0.82–0.84 at every length, so softmax mixing does not rescue trit-rounded values; going from g=64 to g=32 barely helps (≤0.007). 4-bit V is much closer (cosine 0.986–0.992, g=32 better than g=64 by ~0.004) but still misses 0.999 and only saves 1.29–1.32x over all-A8, under the 1.5x bar. Both quantized-V paths are also slower than all-A8 in numpy (trit 0.31–0.39x, A4 0.19–0.23x) because the LUT decode writes a full f32 copy. Error does not grow with length here (unlike ternary K in the prior note), which confirms V error is averaged rather than amplified, but the per-element error of trits on heavy-tailed values is too large to begin with. Keep the KV cache at A8 g=64 for both K and V. Bounded to numpy on synthetic Student-t K/V; real V distributions and a native nibble kernel could move the A4 numbers, not the trit ones.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-10T12:12:30Z (~05:12 MT) wrote pair_A/pair_B **no gap** (roadmap blobs unchanged, JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). `llc_skip_paths`: JuniorHome `5c8319d` (08:17Z), left to LLC. JuniorLLM tip at push time: [07533f8](https://github.com/cloudcover95/JuniorLLM/commit/07533f8) (the prior findings note, not touched here).

## Next question

Since per-element trit error on V is the blocker, does a fixed random orthogonal (Hadamard-style) rotation of each head's 128-dim V and K before quantization, undone on the output, lift A4 g=32 V to cosine ≥ 0.999 at 512–4096 tokens, and what does the rotation cost per decode step versus all-A8?

close: 17c2bc6a4d95444f (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-10_0255)
