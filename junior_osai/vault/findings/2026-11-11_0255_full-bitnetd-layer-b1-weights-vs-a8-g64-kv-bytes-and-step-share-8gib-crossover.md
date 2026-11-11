---
local: true
date: 2026-11-11
wave: "2026-11-11 02:55 (ran ~02:56–03:10 MT; Chief last_run 2026-11-11T09:14:00Z ~02:14 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: viable
tags: [osai, vault, findings, ternary, bitnetd, decode, layer, kv-cache, a8, w158, memory-budget, 8gib, numpy, viable-memory, not-viable-speed]
---

# For a full bitnetd layer at B=1, 2-bit packed W1.58 weights keep the whole model 2.4–3.8x smaller than int8 weights out to T=4096 even after A8 g=64 KV is added, but the per-step LUT expand makes the weight stage 7x slower than int8-resident in numpy

## Question

The prior note [[2026-11-10_1855_rolling-f16-f32-recent-window-plus-a8-g64-kv-vs-all-a8-all-f16-1024-4096]] closed the KV layout question for numpy (keep all-A8 g=64). Its next question: for one full bitnetd layer at B=1 (W1.58 2-bit packed attention projections and FFN at hidden 2560 / FFN 6912, plus A8 g=64 KV at T=1024–4096), what share of resident bytes and of step time goes to weights versus KV, and at what context length does KV overtake packed weights as the binding memory cost on an 8 GiB device? The ternary test inside it: does the 4.00x packed-weight win still give at least a 1.5x smaller total resident footprint (weights + KV) than int8 weights at T up to 4096, and does the packed path cost or save step time? Quant Chief `last_tips.json` `last_run` 2026-11-11T09:14:00Z (~02:14 MT) wrote pair_A/pair_B **no gap**, so this runs the prior note's next question.

## Method

One decoder layer, B=1, hidden 2560, FFN 6912 (gate, up, down), 20 query heads × head_dim 128. Two KV-head settings, both assumed shapes rather than read from a local model config (none is on the box): 20 KV heads (plain multi-head, the shape all prior KV notes used) and 5 KV heads (grouped-query, 4 query heads per KV head). Model totals multiply by an assumed L=30 layers and exclude the embedding and output head.

- Weights: absmean ternary {-1,0,+1} per matrix with one f32 scale. Three resident forms, each timed as the sum of the seven per-step GEMVs: `packed` (4 trits per byte, expanded every step with a 256×4 int8 LUT then f32 GEMV, the I2_S-style resident path), `int8` (one int8 per trit, cast to f32 then GEMV), `f32` (pre-scaled f32, plain GEMV, a memory-heavy ceiling).
- KV: A8 g=64 absmax codes with f16 group scales, decoded to f32 each step; score and V fold as batched matmuls; GQA by grouping query heads per KV head. Cosine is against an f64 attention on unquantized K/V.
- Timing: 2 warm-ups, 7 repeats × 3 passes, median and min–max, run twice back to back on the shared 8-core box (15 GiB RAM), numpy 2.5.4. Seed `202611110255`. A first attempt used an einsum V fold that was 4x slower than the matmul fold at T=4096 MHA (46.5 ms); the script below uses the matmul fold and the tables come from the two reruns.

Commands, from `/workspace/bench0255_1111`: `python3 bench.py results_run1.json`, `python3 bench.py results_run2.json`, then `python3 model.py`.

```python
# One bitnetd-shaped decoder layer at B=1: W1.58 weights (2-bit packed, per-step LUT expand) vs int8-resident vs f32-resident
# weights, plus A8 g64 KV attention at T=1024..4096. Measures per-stage step time and resident bytes; models L-layer totals
# and the KV-vs-weights crossover on an 8 GiB device. Shapes: hidden 2560, FFN 6912, 20 q heads x 128; KV heads 20 (MHA,
# as in prior KV notes) and 5 (GQA variant) - both are assumed shapes, layer count L=30 assumed for the model totals.
import numpy as np, time, json, sys
SEED = 202611110255
Hd, F, NH, D, G = 2560, 6912, 20, 128, 64
Gn = D // G
L = 30
TS = [1024, 2048, 4096]
REP, PASS, WARM = 7, 3, 2
rng = np.random.default_rng(SEED)
LUT = np.array([[((b >> (2 * k)) & 3) for k in range(4)] for b in range(256)], dtype=np.int8) - 1  # code 0,1,2 -> -1,0,+1
def tern(o, i):
    W = rng.standard_normal((o, i)).astype(np.float32); s = np.abs(W).mean()
    T = np.clip(np.rint(W / s), -1, 1).astype(np.int8); return T, np.float32(s)
def pack(T):
    c = (T + 1).astype(np.uint8).reshape(T.shape[0], -1, 4)
    return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).astype(np.uint8)
def gemv_packed(P, s, x):
    W = LUT[P].reshape(P.shape[0], -1)          # per-step expand 2-bit -> int8
    return (W.astype(np.float32) @ x) * s
def gemv_i8(W8, s, x): return (W8.astype(np.float32) @ x) * s
def gemv_f32(Wf, x): return Wf @ x
def q8(X):
    Xg = X.reshape(X.shape[:-1] + (Gn, G))
    sc = (np.abs(Xg).max(-1, keepdims=True) / 127.0).astype(np.float16); sc[sc == 0] = 1
    return np.clip(np.rint(Xg / sc.astype(np.float32)), -127, 127).astype(np.int8).reshape(X.shape), sc[..., 0].astype(np.float32)
def softmax(x):
    m = x.max(-1, keepdims=True); e = np.exp(x - m); return e / e.sum(-1, keepdims=True)
def attn_a8(q, K8, sK, V8, sV, kvh):
    rep = NH // kvh; T = K8.shape[1]; scl = np.float32(1 / np.sqrt(D))
    Kf = K8.astype(np.float32).reshape(kvh, T, Gn, G).transpose(0, 2, 1, 3)
    qg = q.reshape(kvh, rep, Gn, G).transpose(0, 2, 3, 1)                 # (kvh,Gn,G,rep)
    part = np.matmul(Kf, qg)                                               # (kvh,Gn,T,rep)
    s = (part * sK.transpose(0, 2, 1)[..., None]).sum(1) * scl             # (kvh,T,rep)
    p = softmax(s.transpose(0, 2, 1))                                      # (kvh,rep,T)
    Vf = V8.astype(np.float32).reshape(kvh, T, Gn, G).transpose(0, 2, 1, 3)  # (kvh,Gn,T,G)
    w = (p[:, :, :, None] * sV[:, None, :, :]).transpose(0, 3, 1, 2)       # (kvh,Gn,rep,T)
    o = np.matmul(w, Vf)                                                   # (kvh,Gn,rep,G)
    return o.transpose(0, 2, 1, 3).reshape(NH, D)
def bench(f, *a):
    for _ in range(WARM): f(*a)
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter()
        for _ in range(PASS): f(*a)
        ts.append((time.perf_counter() - t0) / PASS * 1e3)
    return float(np.median(ts)), float(min(ts)), float(max(ts))
out = {"seed": SEED, "numpy": np.__version__, "weights": {}, "attn": [], "model": []}
x = rng.standard_normal(Hd).astype(np.float32); xf = rng.standard_normal(F).astype(np.float32)
for kvh in (20, 5):
    shapes = {"q": (Hd, Hd), "k": (kvh * D, Hd), "v": (kvh * D, Hd), "o": (Hd, Hd), "gate": (F, Hd), "up": (F, Hd), "down": (Hd, F)}
    acc = {"packed": [0, 0, 0], "int8": [0, 0, 0], "f32": [0, 0, 0]}; by = {"packed": 0, "int8": 0, "f32": 0}; maxerr = 0.0
    for nm, (o, i) in shapes.items():
        T8, s = tern(o, i); P = pack(T8); Wf = T8.astype(np.float32) * s; v = xf if i == F else x
        ref = Wf.astype(np.float64) @ v.astype(np.float64)
        for tag, f, args, nb in (("packed", gemv_packed, (P, s, v), P.nbytes), ("int8", gemv_i8, (T8, s, v), T8.nbytes), ("f32", gemv_f32, (Wf, v), Wf.nbytes)):
            m, lo, hi = bench(f, *args); acc[tag] = [acc[tag][0] + m, acc[tag][1] + lo, acc[tag][2] + hi]; by[tag] += nb
            maxerr = max(maxerr, float(np.max(np.abs(f(*args) - ref)) / np.max(np.abs(ref))))
        del Wf, T8, P
    out["weights"][str(kvh)] = dict(ms={k: v for k, v in acc.items()}, bytes=by, max_rel_err=maxerr)
    print("kvh", kvh, {k: round(v[0], 3) for k, v in acc.items()}, {k: round(v / 2**20, 2) for k, v in by.items()}, "relerr", maxerr, flush=True)
    for T in TS:
        K = rng.standard_t(4, size=(kvh, T, D)).astype(np.float32) * 0.5; V = rng.standard_t(4, size=(kvh, T, D)).astype(np.float32) * 0.5
        q = rng.standard_normal((NH, D)).astype(np.float32)
        K8, sK = q8(K); V8, sV = q8(V)
        kvb = K8.nbytes + V8.nbytes + 2 * (sK.size + sV.size)
        # reference: f64 GQA attention on unquantized K/V
        rep = NH // kvh; Kr = np.repeat(K, rep, 0).astype(np.float64); Vr = np.repeat(V, rep, 0).astype(np.float64)
        pr = softmax(np.einsum('hd,htd->ht', q.astype(np.float64), Kr) / np.sqrt(D)); REF = np.einsum('ht,htd->hd', pr, Vr)
        oo = attn_a8(q, K8, sK, V8, sV, kvh).astype(np.float64)
        cs = float(oo.ravel() @ REF.ravel() / np.linalg.norm(oo) / np.linalg.norm(REF))
        m, lo, hi = bench(attn_a8, q, K8, sK, V8, sV, kvh)
        out["attn"].append(dict(kvh=kvh, T=T, ms=m, ms_min=lo, ms_max=hi, bytes=int(kvb), cos=cs))
        print(" T", T, round(m, 3), [round(lo, 3), round(hi, 3)], round(kvb / 2**20, 2), "MiB cos", round(cs, 6), flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
```

```python
import json
L = 30; GiB = 2**30
r1, r2 = (json.load(open(f)) for f in ("results_run1.json", "results_run2.json"))
for kvh in ("20", "5"):
    w = r1["weights"][kvh]["bytes"]; kvtok = 2 * int(kvh) * 128 * (1 + 2 / 64)   # per layer per token, A8 codes + f16 g64 scales
    print(f"kvh={kvh} kv_bytes/token/layer={kvtok:.0f}")
    for tag in ("packed", "int8", "f32"):
        Tx = w[tag] / kvtok; fill = (8 * GiB - L * w[tag]) / (L * kvtok)
        print(f"  {tag}: W/layer {w[tag]/2**20:.2f} MiB, model {L*w[tag]/GiB:.3f} GiB, KV overtakes weights at T={Tx:.0f}, 8GiB full at T={fill:.0f}")
    f16 = w["int8"] * 2
    for T in (1024, 2048, 4096, 8192, 16384):
        kv = kvtok * T
        print(f"  T={T}: total packed {L*(w['packed']+kv)/GiB:.3f} GiB, int8 {L*(w['int8']+kv)/GiB:.3f}, f16 {L*(f16+kv)/GiB:.3f}; int8/packed {(w['int8']+kv)/(w['packed']+kv):.2f}x f16/packed {(f16+kv)/(w['packed']+kv):.2f}x; KV share packed {kv/(w['packed']+kv)*100:.1f}%")
    T15 = 2 * (w["int8"] - 1.5 * w["packed"]) / kvtok
    print(f"  int8/packed total ratio falls below 1.5x past T={T15:.0f}")
    for r, n in ((r1, "run1"), (r2, "run2")):
        wm = r["weights"][kvh]["ms"]
        for a in [a for a in r["attn"] if str(a["kvh"]) == kvh]:
            print(f"  {n} T={a['T']}: attn {a['ms']:.3f} ms; attn share of step w/ packed {a['ms']/(a['ms']+wm['packed'][0])*100:.1f}% int8 {a['ms']/(a['ms']+wm['int8'][0])*100:.1f}% f32 {a['ms']/(a['ms']+wm['f32'][0])*100:.1f}%")
```

## Results

Per-layer weight stage (sum of 7 GEMVs; max relative error vs f64 is 4.0e-07 for every form, since all three hold the same trits):

| KV heads | form | MiB per layer | ms run1 / run2 median | vs int8 speed | vs int8 bytes |
|---|---|---|---|---|---|
| 20 | packed | 18.91 | 219.08 / 226.22 | 0.14x | 4.00x smaller |
| 20 | int8 | 75.62 | 31.27 / 33.01 | 1.00x | 1.00x |
| 20 | f32 | 302.50 | 1.368 / 1.365 | 22.9x | 4.00x larger |
| 5 | packed | 16.56 | 203.71 / 200.33 | 0.14x | 4.00x smaller |
| 5 | int8 | 66.25 | 28.41 / 28.75 | 1.00x | 1.00x |
| 5 | f32 | 265.00 | 1.181 / 1.179 | 24.1x | 4.00x larger |

Per-layer A8 g=64 attention step (min–max from run1):

| KV heads | T | KV MiB per layer | ms run1 / run2 (run1 min–max) | cos vs f64 | attention share of step with packed / int8 / f32 weights (run1) |
|---|---|---|---|---|---|
| 20 | 1024 | 5.16 | 1.809 / 1.874 (1.793–1.829) | 0.999935 | 0.8% / 5.5% / 56.9% |
| 20 | 2048 | 10.31 | 3.655 / 3.754 (3.623–3.728) | 0.999902 | 1.6% / 10.5% / 72.8% |
| 20 | 4096 | 20.62 | 11.263 / 11.900 (10.870–13.875) | 0.999938 | 4.9% / 26.5% / 89.2% |
| 5 | 1024 | 1.29 | 0.997 / 1.111 (0.970–1.060) | 0.999934 | 0.5% / 3.4% / 45.8% |
| 5 | 2048 | 2.58 | 1.935 / 2.354 (1.920–2.019) | 0.999901 | 0.9% / 6.4% / 62.1% |
| 5 | 4096 | 5.16 | 3.843 / 4.161 (3.822–3.934) | 0.999923 | 1.9% / 12.6% / 77.9% |

Modeled 30-layer resident totals (weights + A8 g=64 KV; f16 weights are 2x the int8 bytes, computed not timed):

| KV heads | T | packed GiB | int8 GiB | f16 GiB | int8 / packed | f16 / packed | KV share of packed total |
|---|---|---|---|---|---|---|---|
| 20 | 1024 | 0.705 | 2.367 | 4.582 | 3.36x | 6.50x | 21.4% |
| 20 | 2048 | 0.856 | 2.518 | 4.733 | 2.94x | 5.53x | 35.3% |
| 20 | 4096 | 1.158 | 2.820 | 5.035 | 2.43x | 4.35x | 52.2% |
| 20 | 8192 | 1.762 | 3.424 | 5.640 | 1.94x | 3.20x | 68.6% |
| 20 | 16384 | 2.971 | 4.633 | 6.848 | 1.56x | 2.31x | 81.4% |
| 5 | 1024 | 0.523 | 1.979 | 3.920 | 3.78x | 7.49x | 7.2% |
| 5 | 4096 | 0.636 | 2.092 | 4.033 | 3.29x | 6.34x | 23.7% |
| 5 | 16384 | 1.089 | 2.545 | 4.486 | 2.34x | 4.12x | 55.5% |

Crossovers (KV per token per layer is 5,280 bytes with 20 KV heads and 1,320 bytes with 5):

| KV heads | KV overtakes packed weights | KV overtakes int8 weights | 8 GiB filled (packed / int8) | int8/packed total drops under 1.5x |
|---|---|---|---|---|
| 20 | T ≈ 3,755 | T ≈ 15,019 | T ≈ 50,475 / 39,211 | T ≈ 18,773 |
| 5 | T ≈ 13,157 | T ≈ 52,627 | T ≈ 203,761 / 164,290 | T ≈ 65,784 |

The 8 GiB figures count only weights and KV for 30 layers, with no OS, activations, embeddings or runtime overhead, so they are upper bounds. f32-resident weights (8.86 GiB with 20 KV heads) do not fit an 8 GiB device at all.

## Verdict

**Viable on memory, not on speed.** End to end, the ternary weight win survives the KV cache comfortably: at T=4096 the packed model plus A8 KV is 2.43x smaller than int8 weights with 20 KV heads and 3.29x smaller with 5, and 4.35–6.34x smaller than f16, all well past the 1.5x bar at the same 0.9999 attention cosine. With 20 KV heads, KV overtakes the packed weights at about 3.8k tokens, so past that context the KV cache, not the weights, is the binding memory cost; with 5 KV heads that point moves to about 13k. The int8-vs-packed advantage only falls under 1.5x past about 18.8k tokens (20 KV heads) or 65.8k (5). On speed, the packed path loses badly in numpy: expanding 2-bit codes through the LUT every step takes 200–226 ms per layer, 0.14x of int8-resident and about 160x slower than f32-resident, so with packed weights attention is only 0.5–5% of the step and the weight expand is the whole cost. Keep 2-bit packed weights resident for memory and treat the per-step expand as the thing a native kernel has to fix; the KV layout does not matter for step time until weights are fast.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-11T09:14:00Z (~02:14 MT) wrote pair_A/pair_B **no gap** ("02:05 11/11: roadmap blobs unchanged; both pairs no gap"; JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). `llc_skip_paths`: JuniorHome `de17f18` (08:17Z) re-advances landed ROADMAP #5 UE5, left to LLC. JuniorLLM tip at push time: [fffe0bf](https://github.com/cloudcover95/JuniorLLM/commit/fffe0bf) (rails close-hash chain verifier, not under findings and not touched here).

## Next question

The per-step weight expand is now the whole numpy step. Does expanding each packed matrix in row blocks into one reused int8 scratch buffer (no per-step 79M-element allocation, block size 64–1024 rows) bring the packed W1.58 layer to within 1.5x of int8-resident step time while keeping the 4.00x resident-byte win, at B=1 for the full seven-matrix layer?

close: 814d7009c50cf044 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-10_1855)
