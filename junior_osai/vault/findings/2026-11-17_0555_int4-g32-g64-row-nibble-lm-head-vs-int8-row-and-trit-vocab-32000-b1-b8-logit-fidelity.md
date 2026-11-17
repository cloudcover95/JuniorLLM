---
local: true
date: 2026-11-17
wave: "2026-11-17 05:55 (ran ~05:43–05:55 MT; Chief last_run 2026-11-17T12:07:00Z ~05:07 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, lm-head, vocab, logits, int4, nibble, int8, trit, b1, b8, threads, numpy, not-viable]
---

# There is no cheap 1.5x step below int8 on the LM head: a **4-bit g64 head** (nibble pack, f16 scales) is **1.89x smaller than int8 per-row** (43.5 vs 82.0 MB) but keeps top-1 agreement with f32 at only **0.79** at z 3–5 (0.98 at z=8), logit cos 0.9915, KL 1.3e-2, vs int8 per-row 0.94–1.00 / 0.99983 / 2.7e-4. g32 is barely better (top-1 0.78–0.84, KL 9.6e-3 at 1.78x); per-row int4 is close to ternary-bad (0.48–0.52). Every int4 head is also slower than int8 in numpy (threaded 0.65–0.67x at B=1, 0.49–0.50x at B=8 for g64). Keep the LM head at int8 per-row.

## Question

[[2026-11-17_0255_ternary-lm-head-trit-row-trit-g64-vs-int8-row-f16-f32-vocab-32000-b1-b8-logit-fidelity]] found a ternary LM head 3.98x smaller than int8 but broken (top-1 0.29–0.88, cos 0.86). Its next question: is there a middle step — does a **4-bit g64 LM head** keep top-1 ≥ 0.99 and KL ≤ 1e-3 vs f32 at z 3–8 while holding ≥ 1.5x less than int8 per-row, and what does the nibble expand cost at B=1/8?

Quant Chief `last_tips.json` `last_run` 2026-11-17T12:07:00Z (~05:07 MT): pair_A and pair_B both **no gap**, so this note uses the lane's own next question.

## Method

- **Same head and states as 02:55** (seed 202611170255): V=32000, H=2560, Gaussian ×0.02 rows, lognormal row-norm spread σ=0.25, 16 hot input channels ×4; 256 planted-target states per z = 3/5/8. int8_row and trit_row fidelity rows reproduce 02:55 exactly.
- **New formats:** symmetric int4 [−7, 7], absmax per group with f16 scales: `int4_g32` (4.50 bits), `int4_g64` (4.25), `int4_row` (≈4.0). Two nibbles per byte; expand via byte → 2×int8 (u16) LUT `np.take` into a reused f32 tile, group scales applied on the tile, then BLAS. Kernel vs dequantized reference max abs err ≤ 6e-6.
- **Timing:** T=512 single worker or T=256 × 8 worker threads (`OPENBLAS_NUM_THREADS=1`), 5 interleaved rounds per B, 7 repeats + 2 warm, median of round medians with min–max; two full runs. 8 cores, numpy only, offline, nothing bound, no downloads.

Commands (in `/workspace/bench0555_1117`, script sha256 `dbb9e824ff71fe67…`):

```
OPENBLAS_NUM_THREADS=1 python3 lmhead4.py results_run1.json
OPENBLAS_NUM_THREADS=1 python3 lmhead4.py results_run2.json
```

<details><summary>lmhead4.py</summary>

```python
"""LM head 4-bit (05:55 11/17): int4 g64 / int4 g32 / int4 per-row (nibble pack, f16 group scales, byte->2xint8 u16 LUT take)
vs int8 per-row, trit per-row, f32 -- resident bytes, B=1/8 step time, next-token fidelity vs f32. Same synthetic head and
planted-margin states as 02:55 (seed 202611170255). numpy only, offline."""
import numpy as np, time, json, sys
from concurrent.futures import ThreadPoolExecutor
V = 32000; H = 2560; SEED = 202611170255
rng = np.random.default_rng(SEED)
W = rng.standard_normal((V, H), dtype=np.float32) * 0.02
W *= np.exp(rng.normal(0, 0.25, (V, 1))).astype(np.float32)
hot = rng.choice(H, 16, replace=False); W[:, hot] *= 4.0
s8 = (np.abs(W).max(1, keepdims=True) / 127).astype(np.float32); W8 = np.clip(np.rint(W / s8), -127, 127).astype(np.int8)
gr = np.abs(W).mean(1, keepdims=True).astype(np.float32); T_r = np.clip(np.rint(W / gr), -1, 1).astype(np.int8)
def q4(G):  # symmetric int4 [-7,7], absmax per group, f16 scale
    Wg = W.reshape(V, H // G, G)
    s = (np.abs(Wg).max(2, keepdims=True) / 7).astype(np.float16)
    q = np.clip(np.rint(Wg / s.astype(np.float32)), -7, 7).astype(np.int8).reshape(V, H)
    return q, s
Q = {"int4_g32": q4(32), "int4_g64": q4(64), "int4_row": q4(H)}
def pack4(q):
    u = (q.astype(np.int16) & 15).astype(np.uint8).reshape(V, -1, 2)
    return (u[..., 0] | (u[..., 1] << 4)).astype(np.uint8)
def pack2(T):
    c = np.where(T < 0, 2, T).astype(np.uint8).reshape(V, -1, 4)
    return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).astype(np.uint8)
P4 = {k: pack4(q) for k, (q, s) in Q.items()}; P_r = pack2(T_r)
codes = np.arange(256, dtype=np.uint8)
n4 = np.stack([codes & 15, codes >> 4], 1).astype(np.int16); n4[n4 > 7] -= 16
LUT_U16 = np.ascontiguousarray(n4.astype(np.int8)).view(np.uint16).ravel()
lt = np.stack([(codes >> (2 * k)) & 3 for k in range(4)], 1).astype(np.int8); lt[lt == 2] = -1
LUT_U32 = np.ascontiguousarray(lt).view(np.uint32).ravel()
for k, (q, s) in Q.items(): assert np.array_equal(LUT_U16[P4[k]].view(np.int8).reshape(V, H), q)
BYTES = {"f32": W.nbytes, "int8_row": W8.nbytes + s8.nbytes, **{k: P4[k].nbytes + Q[k][1].nbytes for k in Q},
         "trit_row": P_r.nbytes + gr.nbytes}
REF = {"f32": W, "int8_row": W8 * s8, "trit_row": T_r * gr,
       **{k: (q.reshape(V, s.shape[1], -1) * s.astype(np.float32)).reshape(V, H) for k, (q, s) in Q.items()}}
def make(kind, T, nw):
    pool = ThreadPoolExecutor(nw) if nw > 1 else None; scr = {}
    def tile(x, y, o0, n, wid):
        if kind == "f32":
            np.matmul(x, W[o0:o0+n].T, out=y[:, o0:o0+n]); return
        s = scr.setdefault((wid, "f"), np.empty((T, H), np.float32))[:n]
        if kind == "int8_row":
            np.copyto(s, W8[o0:o0+n], casting="unsafe")
        elif kind == "trit_row":
            u = scr.setdefault((wid, "u"), np.empty((T, H // 4), np.uint32))[:n]
            np.take(LUT_U32, P_r[o0:o0+n], out=u); np.copyto(s, u.view(np.int8), casting="unsafe")
        else:
            u = scr.setdefault((wid, "h"), np.empty((T, H // 2), np.uint16))[:n]
            np.take(LUT_U16, P4[kind][o0:o0+n], out=u); np.copyto(s, u.view(np.int8), casting="unsafe")
            sc = Q[kind][1][o0:o0+n]
            if sc.shape[1] > 1:
                sv = s.reshape(n, sc.shape[1], -1); sv *= sc.astype(np.float32)
        np.matmul(x, s.T, out=y[:, o0:o0+n])
        if kind == "int8_row": y[:, o0:o0+n] *= s8[o0:o0+n, 0]
        elif kind == "trit_row": y[:, o0:o0+n] *= gr[o0:o0+n, 0]
        elif kind == "int4_row": y[:, o0:o0+n] *= Q[kind][1][o0:o0+n, 0, 0].astype(np.float32)
    def mm(x):
        y = np.empty((x.shape[0], V), np.float32)
        if kind == "f32" and pool is None: np.matmul(x, W.T, out=y); return y
        starts = list(range(0, V, T))
        if pool is None:
            for o0 in starts: tile(x, y, o0, min(T, V - o0), 0)
        else:
            ch = [starts[w::nw] for w in range(nw)]
            def run(w):
                for o0 in ch[w]: tile(x, y, o0, min(T, V - o0), w)
            list(pool.map(run, range(nw)))
        return y
    return mm
def states(n, z):
    h = rng.standard_normal((n, H), dtype=np.float32); t = rng.integers(0, V, n)
    sd = (h @ W.T).std(1); w = W[t]; nn = (w * w).sum(1)
    h += ((z * sd)[:, None] * w / nn[:, None]).astype(np.float32); return h
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def fidelity():
    out = {}
    for z in (3.0, 5.0, 8.0):
        h = states(256, z); l0 = h @ W.T; p0 = softmax(l0); top0 = np.argsort(-l0, 1)[:, :5]
        for k in ["int8_row", "int4_g32", "int4_g64", "int4_row", "trit_row"]:
            l = h @ REF[k].T; p = softmax(l); top = np.argsort(-l, 1)[:, :5]
            cos = (l * l0).sum(1) / np.linalg.norm(l, axis=1) / np.linalg.norm(l0, axis=1)
            kl = (p0 * (np.log(p0 + 1e-30) - np.log(p + 1e-30))).sum(1)
            out[f"z{z:g}_{k}"] = dict(top1=float((top[:, 0] == top0[:, 0]).mean()),
                top5=float(np.mean([len(set(a) & set(b)) / 5 for a, b in zip(top, top0)])),
                cos=float(np.median(cos)), kl=float(np.median(kl)))
    return out
def timeit(f, x, rep=7, warm=2):
    for _ in range(warm): f(x)
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter(); f(x); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))
def main(path):
    xc = rng.standard_normal((4, H), dtype=np.float32)
    errs = {k: float(np.abs(make(k, 512, 1)(xc) - xc @ REF[k].T).max()) for k in BYTES}
    fid = fidelity()
    cfgs = [("f32", 0, 1)] + [(k, T, nw) for k in ["f32", "int8_row", "int4_g32", "int4_g64", "int4_row", "trit_row"] for T, nw in ((512, 1), (256, 8))]
    fns = {c: make(*c) for c in cfgs}; tim = {}
    for B in (1, 8):
        x = rng.standard_normal((B, H), dtype=np.float32); rounds = {c: [] for c in cfgs}
        for r in range(5):
            order = cfgs[r % len(cfgs):] + cfgs[:r % len(cfgs)]
            for c in order: rounds[c].append(timeit(fns[c], x))
        for c in cfgs:
            v = rounds[c]; tim[f"B{B}_{c[0]}_T{c[1]}_W{c[2]}"] = [float(np.median(v)), float(min(v)), float(max(v))]
    json.dump(dict(V=V, H=H, bytes=BYTES, kernel_maxerr=errs, fidelity=fid, timing=tim), open(path, "w"), indent=1)
    print(json.dumps(dict(bytes=BYTES, kernel_maxerr=errs), indent=1))
    for k, v in fid.items(): print(k, v)
    for k, v in tim.items(): print(k, "%.2f (%.2f-%.2f)" % tuple(v))
if __name__ == "__main__":
    main(sys.argv[1])
```

</details>

## Results

### Resident bytes (V=32000)

| format | bytes | MB | bits/weight | int8_row / x |
|---|---:|---:|---:|---:|
| `f32` | 327,680,000 | 327.7 | 32.000 | 0.25x |
| `int8_row` | 82,048,000 | 82.0 | 8.012 | 1.00x |
| `int4_g32` | 46,080,000 | 46.1 | 4.500 | 1.78x |
| `int4_g64` | 43,520,000 | 43.5 | 4.250 | 1.89x |
| `int4_row` | 41,024,000 | 41.0 | 4.006 | 2.00x |
| `trit_row` | 20,608,000 | 20.6 | 2.013 | 3.98x |

### Next-token fidelity vs f32 (deterministic, identical in both runs)

| z | format | top-1 | top-5 overlap | logit cos | KL |
|---|---|---:|---:|---:|---:|
| 3 | `int8_row` | 0.938 | 0.973 | 0.99983 | 2.70e-04 |
| 3 | `int4_g32` | 0.777 | 0.823 | 0.99399 | 9.62e-03 |
| 3 | `int4_g64` | 0.789 | 0.798 | 0.99145 | 1.36e-02 |
| 3 | `int4_row` | 0.477 | 0.544 | 0.94783 | 9.14e-02 |
| 3 | `trit_row` | 0.316 | 0.359 | 0.86612 | 2.63e-01 |
| 5 | `int8_row` | 0.980 | 0.974 | 0.99983 | 2.74e-04 |
| 5 | `int4_g32` | 0.840 | 0.870 | 0.99413 | 9.60e-03 |
| 5 | `int4_g64` | 0.789 | 0.824 | 0.99164 | 1.37e-02 |
| 5 | `int4_row` | 0.516 | 0.598 | 0.94876 | 9.28e-02 |
| 5 | `trit_row` | 0.293 | 0.416 | 0.86238 | 2.83e-01 |
| 8 | `int8_row` | 1.000 | 0.976 | 0.99983 | 2.66e-04 |
| 8 | `int4_g32` | 0.984 | 0.865 | 0.99421 | 9.39e-03 |
| 8 | `int4_g64` | 0.984 | 0.845 | 0.99178 | 1.31e-02 |
| 8 | `int4_row` | 0.941 | 0.645 | 0.94942 | 8.87e-02 |
| 8 | `trit_row` | 0.863 | 0.485 | 0.85912 | 5.35e-01 |

### Step time, ms (median of rounds, min–max)

| B | config | run1 | run2 | int8_row (same threading) / x |
|---|---|---:|---:|---:|
| 1 | `f32_T0_W1` | 21.76 (21.12–22.58) | 21.37 (20.62–22.46) | 1.25 / 1.33 |
| 1 | `f32_T512_W1` | 20.43 (18.41–22.10) | 21.35 (20.06–22.14) | 1.33 / 1.33 |
| 1 | `f32_T256_W8` | 21.74 (20.20–24.82) | 24.08 (22.35–24.80) | 0.73 / 0.69 |
| 1 | `int8_row_T512_W1` | 27.13 (25.07–28.92) | 28.47 (26.78–29.48) | 1.00 / 1.00 |
| 1 | `int8_row_T256_W8` | 15.90 (15.24–18.09) | 16.62 (15.92–16.96) | 1.00 / 1.00 |
| 1 | `int4_g32_T512_W1` | 98.51 (96.08–101.32) | 99.02 (98.15–100.48) | 0.28 / 0.29 |
| 1 | `int4_g32_T256_W8` | 25.93 (24.25–29.64) | 26.09 (24.40–27.32) | 0.61 / 0.64 |
| 1 | `int4_g64_T512_W1` | 94.53 (91.63–95.92) | 96.28 (93.14–98.94) | 0.29 / 0.30 |
| 1 | `int4_g64_T256_W8` | 24.58 (23.74–27.70) | 24.74 (24.35–26.75) | 0.65 / 0.67 |
| 1 | `int4_row_T512_W1` | 70.05 (67.08–73.07) | 70.64 (70.03–78.87) | 0.39 / 0.40 |
| 1 | `int4_row_T256_W8` | 22.62 (21.83–24.66) | 24.74 (22.28–26.93) | 0.70 / 0.67 |
| 1 | `trit_row_T512_W1` | 52.31 (50.68–52.51) | 54.62 (50.01–61.37) | 0.52 / 0.52 |
| 1 | `trit_row_T256_W8` | 23.28 (20.72–26.23) | 21.84 (20.80–23.32) | 0.68 / 0.76 |
| 8 | `f32_T0_W1` | 42.50 (39.91–48.01) | 44.70 (41.94–46.73) | 1.13 / 1.08 |
| 8 | `f32_T512_W1` | 42.32 (38.55–48.90) | 44.57 (42.04–47.29) | 1.13 / 1.08 |
| 8 | `f32_T256_W8` | 5.95 (5.39–7.82) | 6.98 (5.90–8.30) | 1.46 / 1.50 |
| 8 | `int8_row_T512_W1` | 48.01 (46.05–54.90) | 48.24 (47.18–52.31) | 1.00 / 1.00 |
| 8 | `int8_row_T256_W8` | 8.70 (7.20–11.08) | 10.49 (8.95–11.31) | 1.00 / 1.00 |
| 8 | `int4_g32_T512_W1` | 116.01 (114.94–117.18) | 120.25 (116.13–125.13) | 0.41 / 0.40 |
| 8 | `int4_g32_T256_W8` | 21.14 (19.81–41.50) | 19.31 (18.73–22.36) | 0.41 / 0.54 |
| 8 | `int4_g64_T512_W1` | 113.51 (111.29–124.53) | 118.98 (110.55–120.53) | 0.42 / 0.41 |
| 8 | `int4_g64_T256_W8` | 17.79 (17.29–21.38) | 20.83 (18.49–22.93) | 0.49 / 0.50 |
| 8 | `int4_row_T512_W1` | 88.72 (87.85–90.03) | 91.48 (88.79–95.58) | 0.54 / 0.53 |
| 8 | `int4_row_T256_W8` | 15.07 (14.66–16.45) | 17.03 (14.73–17.95) | 0.58 / 0.62 |
| 8 | `trit_row_T512_W1` | 71.98 (70.05–74.85) | 74.42 (69.76–78.96) | 0.67 / 0.65 |
| 8 | `trit_row_T256_W8` | 12.50 (12.03–14.09) | 14.13 (12.07–14.65) | 0.70 / 0.74 |

## Verdict

**not-viable.** int4 g32/g64 clears the memory bar (1.78x / 1.89x less than int8 per-row) but misses the fidelity bar by a wide margin: top-1 0.78–0.84 at typical margins and KL ~1e-2, about 35–50x int8's KL. At 2560 terms per logit, int4 rounding noise (cos 0.991–0.994) still flips the argmax for 1 in 5 tokens. Per-row int4 (cos 0.948) sits closer to ternary than to int8. Speed does not help either: the nibble LUT take plus the per-group scale multiply makes int4 g64 0.65–0.67x of int8 at B=1 and about 0.5x at B=8 threaded, slower even than the ternary head. Together with 02:55: ternary/int4 belong on the hidden-layer weights; the vocab projection stays int8 per-row (82 MB at V=32000, ~329 MB at V=128256).

**Caveat:** synthetic Gaussian head and planted-margin states; real heads with tied embeddings or stronger outliers may differ (likely worse for int4, since absmax groups are outlier-sensitive); numpy on a loaded 8-core box (~13 of 15 GB used by other processes).

## Pair files this wave

At Quant Chief 05:05 MT (`last_run` 12:07Z Nov 17), pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave.
- Roadmap blobs unchanged (JuniorLLM docs/BETA_TO_OS.md bac6923, JuniorHome docs/ROADMAP.md 4efb666).
- JuniorLLM head before this push was 902c348 (02:51 MT, cloudcover95, no roadmap item: adds `rails/ratepack/rate_pack_check.py` only), parent e9cd0c1 (the 02:55 note). No Trit Pack, Handshake or LLC commit touched `junior_osai/vault/findings/` in the last 6 hours; Chief lists no LLC skip paths this wave.
- This note adds one file under `junior_osai/vault/findings/`.

## Next question

The LM head is settled at int8 per-row, so return to batched decode on the hidden layers: at B=8/16, does **sharing one u32-LUT expanded tile across a fused gate|up|down FFN pass** (expand once per tile, reuse for the B tokens and both gate/up halves) beat the 05:55 11/16 threaded tiled expand (B=8 14.3 ms) by ≥ 1.5x at the same 26.3 MB resident, exact W1.58?

close: sha256(prev || this)[:16] = eb149db58106ca83
