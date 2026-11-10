---
local: true
date: 2026-11-10
wave: "2026-11-10 10:55 (ran ~10:55–11:05 MT; Chief last_run 2026-11-10T17:14:30Z ~10:14 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, attention, kv-cache, hadamard, rotation, a4, a8, trit, numpy, not-viable]
---

# Hadamard rotation lifts 4-bit KV but not to cosine 0.999: rotated A4 V 0.995, rotated K4+V4 0.989–0.993 at 1.83x less memory than all-A8; rotated trit V 0.88–0.90

## Question

Prior note [[2026-11-10_0555_value-only-kv-trit-a4-g32-g64-v-with-a8-k-vs-all-a8-f16-attention-512-4096]] found per-element quant error on V is the blocker (A4 g=32 V cos 0.986–0.992, trit V 0.82–0.84). Does a fixed orthogonal 128-dim Walsh-Hadamard rotation of each head's K and V before per-group quantization (q rotated the same way so scores are unchanged, output rotated back) lift A4 g=32 V, or A4 g=32 K and V together, or trit g=32 V, to cosine ≥ 0.999 at 512–4096 cached tokens, and what does the rotation cost per decode step and per appended token versus all-A8 g=64? Quant Chief `last_tips.json` `last_run` 2026-11-10T17:14:30Z (~10:14 MT) wrote pair_A/pair_B **no gap**, so this runs the prior note's next question.

## Method

Same bitnetd-shaped attention decode step as the two prior notes: B=1, 20 heads × head_dim 128 (hidden 2560), cached length T ∈ {512, 1024, 2048, 4096}, K Student-t(4)×0.5 with 6 outlier channels ×6, V Student-t(4)×0.5, q standard normal, seed `202611101055`. Error is cosine and mae of the attention output against an f64 reference on unquantized, unrotated K/V (deterministic). Rotation R is the normalized 128×128 Sylvester Hadamard (orthogonal and symmetric), applied as `K @ R`, `V @ R`, `q @ R` and `out @ R`. Quantizers: A8 g=64 absmax ±127, A4 g=32 absmax ±7 packed two nibbles per byte (unpacked with bit ops this time, no gather), trit g=32 absmean packed four per byte (256×4 LUT). All paths use the same matmul-based score and scale-folded V contraction, so this run's all-A8 baseline is faster than the prior note's einsum baseline. Bytes are stored K+V codes plus f16 scales. Append cost is quantizing (and for the rotated path, rotating) one new k and v for all 20 heads. Timing: 2 warm-ups, 7 repeats × 3 passes, median and min–max; the whole script was run twice back to back on the shared 8-core box, numpy 2.5.4.

Command, from `/workspace/bench1055_1110`: `python3 bench.py results_run1.json` then `python3 bench.py results_run2.json`.

```python
# Hadamard-rotated KV quant for bitnetd attention step (B=1 decode): rotate K,V (and q) by normalized 128-dim Walsh-Hadamard before
# per-group quant; un-rotate output. Paths: all-A8 g64 baseline, K8+V4g32 plain/rotated, K4g32+V4g32 rotated, K8+Vtrit g32 rotated.
import numpy as np, time, json, sys
SEED = 202611101055
H, D = 20, 128
TS = [512, 1024, 2048, 4096]
REP, PASS, WARM = 7, 3, 2
rng = np.random.default_rng(SEED)
def hadamard(n):
    M = np.array([[1.0]])
    while M.shape[0] < n: M = np.block([[M, M], [M, -M]])
    return (M / np.sqrt(n)).astype(np.float32)
R = hadamard(D)  # orthogonal, symmetric: R @ R = I
TLUT = (np.array([[((b >> (2*i)) & 3) for i in range(4)] for b in range(256)], np.int8) - 1).astype(np.float32)

def make_kv(T):
    K = rng.standard_t(4, size=(H, T, D)).astype(np.float32) * 0.5
    ch = rng.choice(D, 6, replace=False); K[:, :, ch] *= 6.0
    V = rng.standard_t(4, size=(H, T, D)).astype(np.float32) * 0.5
    q = rng.standard_normal((H, D)).astype(np.float32)
    return q, K, V

def q8(X, G=64):
    Xg = X.reshape(H, -1, D // G, G)
    s = (np.abs(Xg).max(-1, keepdims=True) / 127.0).astype(np.float16); s[s == 0] = 1
    return np.clip(np.rint(Xg / s.astype(np.float32)), -127, 127).astype(np.int8).reshape(X.shape), s[..., 0]
def q4(X, G=32):
    Xg = X.reshape(H, -1, D // G, G)
    s = (np.abs(Xg).max(-1, keepdims=True) / 7.0).astype(np.float16); s[s == 0] = 1
    c = (np.clip(np.rint(Xg / s.astype(np.float32)), -7, 7).astype(np.int16) + 8).astype(np.uint8).reshape(H, X.shape[1], D // 2, 2)
    return (c[..., 0] | (c[..., 1] << 4)), s[..., 0]
def qt(X, G=32):
    Xg = X.reshape(H, -1, D // G, G)
    s = np.abs(Xg).mean(-1, keepdims=True).astype(np.float16); s[s == 0] = 1
    t = (np.clip(np.rint(Xg / s.astype(np.float32)), -1, 1).astype(np.int8) + 1).astype(np.uint8).reshape(H, X.shape[1], D // 4, 4)
    return (t[..., 0] | (t[..., 1] << 2) | (t[..., 2] << 4) | (t[..., 3] << 6)), s[..., 0]

def dec8(c): return c.astype(np.float32)
def dec4(p):  # bit-op nibble unpack (no gather)
    out = np.empty(p.shape[:-1] + (D,), np.float32)
    out[..., 0::2] = (p & 15); out[..., 1::2] = (p >> 4); out -= 8; return out
def dect(p): return TLUT[p].reshape(p.shape[:-1] + (D,))

def softmax(x):
    m = x.max(-1, keepdims=True); e = np.exp(x - m); return e / e.sum(-1, keepdims=True)
sc = np.float32(1.0 / np.sqrt(D))
def scores(q, Kf, sK):  # Kf decoded codes (H,T,D), sK (H,T,Gn)
    T = Kf.shape[1]; Gn = sK.shape[-1]; G = D // Gn
    part = np.matmul(Kf.reshape(H, T, Gn, G).transpose(0, 2, 1, 3), q.reshape(H, Gn, G, 1))[..., 0]  # (H,Gn,T)
    return softmax((part.transpose(0, 2, 1) * sK.astype(np.float32)).sum(-1) * sc)
def vfold(p, Vf, sV):
    T = p.shape[1]; Gn = sV.shape[-1]; G = D // Gn
    w = (p[:, :, None] * sV.astype(np.float32)).transpose(0, 2, 1)[:, :, None, :]  # (H,Gn,1,T)
    return np.matmul(w, Vf.reshape(H, T, Gn, G).transpose(0, 2, 1, 3))[:, :, 0, :].reshape(H, D)
def make_step(dK, dV, rot):
    def f(q, Kc, sK, Vc, sV):
        qq = q @ R if rot else q
        o = vfold(scores(qq, dK(Kc), sK), dV(Vc), sV)
        return o @ R if rot else o
    return f

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
nb = lambda *xs: int(sum(x.nbytes for x in xs))
# per-token append cost: rotate+quant one new (H,1,D) k and v
kn = rng.standard_normal((H, 1, D)).astype(np.float32); vn = rng.standard_normal((H, 1, D)).astype(np.float32)
app = {"append_A8_plain": lambda: (q8(kn), q8(vn)), "append_K4V4_rot": lambda: (q4(kn @ R), q4(vn @ R))}
for n, f in app.items():
    med, lo, hi = bench(f); out["rows"].append(dict(T=1, path=n, ms=med, ms_min=lo, ms_max=hi)); print(f"{n} med={med*1e3:.1f}us [{lo*1e3:.1f}-{hi*1e3:.1f}]")
for T in TS:
    q, K, V = make_kv(T); ref = ref64(q, K, V)
    Kr, Vr = K @ R, V @ R
    K8, sK8 = q8(K); V8, sV8 = q8(V)
    V4, sV4 = q4(V); V4r, sV4r = q4(Vr); K8r, sK8r = q8(Kr); K4r, sK4r = q4(Kr); Vtr, sVtr = qt(Vr); K4, sK4 = q4(K)
    paths = {
        "K8_V8g64": (make_step(dec8, dec8, False), (q, K8, sK8, V8, sV8)),
        "K8_V4g32": (make_step(dec8, dec4, False), (q, K8, sK8, V4, sV4)),
        "rK8_rV4g32": (make_step(dec8, dec4, True), (q, K8r, sK8r, V4r, sV4r)),
        "K4_V4g32": (make_step(dec4, dec4, False), (q, K4, sK4, V4, sV4)),
        "rK4_rV4g32": (make_step(dec4, dec4, True), (q, K4r, sK4r, V4r, sV4r)),
        "rK8_rVtg32": (make_step(dec8, dect, True), (q, K8r, sK8r, Vtr, sVtr)),
    }
    for name, (f, args) in paths.items():
        o = f(*args); med, lo, hi = bench(f, *args); by = nb(*args[1:])
        r = dict(T=T, path=name, bytes=by, ms=med, ms_min=lo, ms_max=hi, cos=cos(o, ref), mae=float(np.mean(np.abs(o - ref))))
        out["rows"].append(r)
        print(f"T={T:5d} {name:11s} bytes={by:>9d} med={med:8.3f}ms [{lo:.3f}-{hi:.3f}] cos={r['cos']:.6f} mae={r['mae']:.2e}", flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
```

## Results

Decode step, B=1:

| T | path | K+V MiB | mem vs f16 | mem vs all-A8 | ms run1 / run2 median (run1 min–max) | speed vs all-A8 run1 / run2 | cosine | mae |
|---|---|---|---|---|---|---|---|---|
| 512 | K8_V8g64 | 2.58 | 1.94x | 1.00x | 1.205 / 1.140 (1.178–1.377) | 1.00x / 1.00x | 0.999610 | 1.65e-03 |
| 512 | K8_V4g32 | 1.99 | 2.51x | 1.29x | 1.816 / 1.812 (1.757–3.283) | 0.66x / 0.63x | 0.990233 | 9.04e-03 |
| 512 | rK8_rV4g32 | 1.99 | 2.51x | 1.29x | 1.860 / 1.762 (1.769–1.957) | 0.65x / 0.65x | 0.994863 | 6.40e-03 |
| 512 | K4_V4g32 | 1.41 | 3.56x | 1.83x | 2.541 / 2.325 (2.402–2.637) | 0.47x / 0.49x | 0.964722 | 1.57e-02 |
| 512 | rK4_rV4g32 | 1.41 | 3.56x | 1.83x | 2.454 / 2.375 (2.392–2.572) | 0.49x / 0.48x | 0.990173 | 9.05e-03 |
| 512 | rK8_rVtg32 | 1.68 | 2.98x | 1.53x | 4.253 / 4.183 (4.172–4.494) | 0.28x / 0.27x | 0.883479 | 3.23e-02 |
| 1024 | K8_V8g64 | 5.16 | 1.94x | 1.00x | 2.219 / 2.231 (2.182–2.404) | 1.00x / 1.00x | 0.999753 | 1.34e-03 |
| 1024 | K8_V4g32 | 3.98 | 2.51x | 1.29x | 3.446 / 3.560 (3.260–4.246) | 0.64x / 0.63x | 0.989742 | 8.60e-03 |
| 1024 | rK8_rV4g32 | 3.98 | 2.51x | 1.29x | 3.438 / 3.589 (3.283–4.115) | 0.65x / 0.62x | 0.995157 | 5.92e-03 |
| 1024 | K4_V4g32 | 2.81 | 3.56x | 1.83x | 4.877 / 4.858 (4.676–5.459) | 0.46x / 0.46x | 0.928163 | 1.86e-02 |
| 1024 | rK4_rV4g32 | 2.81 | 3.56x | 1.83x | 4.738 / 5.019 (4.604–4.886) | 0.47x / 0.44x | 0.988916 | 9.59e-03 |
| 1024 | rK8_rVtg32 | 3.36 | 2.98x | 1.53x | 8.264 / 8.910 (8.174–8.607) | 0.27x / 0.25x | 0.899239 | 2.89e-02 |
| 2048 | K8_V8g64 | 10.31 | 1.94x | 1.00x | 4.369 / 4.492 (4.339–4.570) | 1.00x / 1.00x | 0.999814 | 3.18e-03 |
| 2048 | K8_V4g32 | 7.97 | 2.51x | 1.29x | 6.727 / 6.891 (6.659–7.959) | 0.65x / 0.65x | 0.990793 | 2.23e-02 |
| 2048 | rK8_rV4g32 | 7.97 | 2.51x | 1.29x | 6.825 / 6.969 (6.757–8.557) | 0.64x / 0.64x | 0.995505 | 1.54e-02 |
| 2048 | K4_V4g32 | 5.62 | 3.56x | 1.83x | 9.443 / 9.619 (8.986–11.367) | 0.46x / 0.47x | 0.984317 | 3.32e-02 |
| 2048 | rK4_rV4g32 | 5.62 | 3.56x | 1.83x | 9.671 / 9.692 (9.509–10.052) | 0.45x / 0.46x | 0.993301 | 2.08e-02 |
| 2048 | rK8_rVtg32 | 6.72 | 2.98x | 1.53x | 16.446 / 16.426 (16.106–16.926) | 0.27x / 0.27x | 0.899323 | 7.54e-02 |
| 4096 | K8_V8g64 | 20.62 | 1.94x | 1.00x | 12.874 / 12.337 (12.714–13.868) | 1.00x / 1.00x | 0.999808 | 1.83e-03 |
| 4096 | K8_V4g32 | 15.94 | 2.51x | 1.29x | 19.247 / 18.142 (18.308–22.330) | 0.67x / 0.68x | 0.992124 | 1.01e-02 |
| 4096 | rK8_rV4g32 | 15.94 | 2.51x | 1.29x | 20.166 / 18.699 (18.942–27.292) | 0.64x / 0.66x | 0.995350 | 7.61e-03 |
| 4096 | K4_V4g32 | 11.25 | 3.56x | 1.83x | 25.728 / 26.105 (25.377–26.194) | 0.50x / 0.47x | 0.969010 | 2.05e-02 |
| 4096 | rK4_rV4g32 | 11.25 | 3.56x | 1.83x | 25.949 / 25.450 (24.826–26.660) | 0.50x / 0.48x | 0.990876 | 1.17e-02 |
| 4096 | rK8_rVtg32 | 13.44 | 2.98x | 1.53x | 40.253 / 40.014 (37.043–54.305) | 0.32x / 0.31x | 0.883347 | 3.76e-02 |

Per appended token (all 20 heads, k and v):

| path | run1 / run2 median (run1 min–max) |
|---|---|
| append_A8_plain | 34.0 / 29.7 µs (32.2–39.1) |
| append_K4V4_rot | 90.5 / 73.4 µs (82.1–95.0) |

Error columns are identical across runs (same seed). `r` = Hadamard-rotated, `K8`/`V8` = A8 g=64, `K4`/`V4` = A4 g=32, `Vt` = trit g=32.

## Verdict

**Not viable.** Rotation helps every 4-bit path but none reaches 0.999. For A4 g=32 V with A8 K it lifts cosine from 0.990–0.992 to about 0.995, still only 1.29x less memory than all-A8 and 0.6–0.7x its speed. The interesting case is rotated K4+V4 g=32: it clears the 1.5x memory bar (1.83x less than all-A8, 3.56x less than f16) and rotation rescues the outlier-channel K (unrotated K4 0.93–0.98, rotated 0.989–0.993), but it still misses 0.999 and runs at 0.44–0.50x of all-A8 in numpy. Rotated trit V improves on the unrotated trit (0.82–0.84 to 0.88–0.90) and is far from usable. Rotation itself is cheap: the per-step q and output rotations are lost in the noise next to the contraction (rotated and unrotated K8+V4 run within a few percent of each other), and appending a rotated 4-bit token costs about 73–91 µs versus 30–34 µs for plain A8, which is small against a 1–13 ms step. Keep the KV cache at A8 g=64 for K and V; ternary stays a weight-memory win only. Bounded to numpy on synthetic Student-t K/V; real activations with stronger channel outliers may gain more from rotation, and a native nibble kernel would change the speed column but not the cosine.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-10T17:14:30Z (~10:14 MT) wrote pair_A/pair_B **no gap** (roadmap blobs unchanged, JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). `llc_skip_paths`: JuniorHome `d6f1c0d` (16:19Z) ROADMAP #3 i2sd, left to LLC. JuniorLLM tip at push time: [7b33d74](https://github.com/cloudcover95/JuniorLLM/commit/7b33d74) (the prior findings note, not touched here).

## Next question

The KV cache line looks settled at A8 g=64; does the remaining decode budget sit in the f32 softmax and V contraction rather than the stored bytes? Measure, at T=512–4096, how the all-A8 g=64 step splits between dequant, score matmul, softmax, and V fold, and whether a fused int8 score (int32 accumulate, one scale per group) beats the f32-decoded score path by 1.5x in numpy.

close: 9dd921b021b485a3 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-10_0555)
