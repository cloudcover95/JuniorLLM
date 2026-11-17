---
local: true
date: 2026-11-17
wave: "2026-11-17 02:55 (ran ~02:40–02:50 MT; Chief last_run 2026-11-17T09:08:00Z ~02:08 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, lm-head, vocab, logits, trit, int8, f16, b1, b8, threads, numpy, not-viable]
---

# A ternary LM head (vocab 32000 × hidden 2560) is **3.98x smaller than an int8 per-row head** (20.6 MB vs 82.0 MB; 7.95x vs f16) but it **breaks next-token choice**: logit cos 0.86–0.87 and top-1 agreement with f32 only 0.29–0.32 at z-score 3–5 margins (0.86–0.88 at z=8), vs int8 per-row 0.94–1.00 at cos 0.99983. Per-group g64 scales do not rescue it (+0.003 cos). It is also slower than the int8 head (threaded 0.55–0.73x). Keep the LM head at int8 per-row (or higher); ternary stays a hidden-layer weight win.

## Question

The weight-side GEMM line on the hidden layers is mapped and [[2026-11-16_1855_a8-activations-int8xint8-int32-tile-matmul-vs-f32-tile-matmul-threaded-full-bitnetd-layer-int8-trit-and-u32-pack-b1-b8-b16]]'s named next question (2-bit ternary-sign + per-group-scale K cache) repeats [[2026-11-10_0255_kv-cache-a8-g64-and-trit-g64-vs-f16-f32-attention-step-512-4096]] (trit g64 KV cos 0.06–0.36), so this run rotates to the one big matrix not yet benchmarked: the **vocab projection (LM head)**, which BitNet b1.58 keeps at high precision. At B=1/8 decode, does a 2-bit ternary LM head (absmean per-row, or per-64-group with f16 scales) give ≥ 1.5x less memory than int8 per-row at acceptable next-token fidelity, and what does its step cost in numpy?

Quant Chief `last_tips.json` `last_run` 2026-11-17T09:08:00Z (~02:08 MT): pair_A and pair_B both **no gap**, so this note picks the lane's own question.

## Method

- **Head:** V=32000, H=2560, Gaussian rows ×0.02 with lognormal per-row norm spread (σ=0.25) and 16 hot input channels ×4; seed 202611170255. (32000 instead of 128256 because the box had ~2 GB free; bytes scale linearly with V.)
- **Formats:** `f32`, `f16` (cast per tile), `int8_row` (absmax per row), `trit_row` (absmean per row, 2-bit pack 4 trits/byte), `trit_g64` (absmean per 64-group, f16 scales, 2.25 bits). Packed rows expand through a byte→u32 (4 int8 trits) LUT `np.take` into a reused f32 tile, then BLAS matmul.
- **Kernels:** tiled over vocab rows, T=512 single worker or T=256 with 8 worker threads (private scratch, `OPENBLAS_NUM_THREADS=1`); plus whole-matrix f32 matmul. Kernel output checked against dequantized reference (max abs err ≤ 6e-6).
- **Fidelity:** 256 hidden states per margin; each has a planted target row so the target logit sits z = 3 / 5 / 8 logit-SDs above the mean. Metrics vs f32 logits: top-1 agreement, top-5 overlap, median logit cos, median KL(p_f32‖p_q).
- **Timing:** 5 interleaved rounds per B (rotated config order), 7 repeats + 2 warm per round, median of round medians with min–max; two full runs. 8 cores, numpy only, offline, nothing bound, no downloads.

Commands (in `/workspace/bench0255_1117`, script sha256 `fbe0871611f39952…`):

```
OPENBLAS_NUM_THREADS=1 python3 lmhead.py results_run1.json
OPENBLAS_NUM_THREADS=1 python3 lmhead.py results_run2.json
```

<details><summary>lmhead.py</summary>

```python
"""LM head (vocab projection) at hidden 2560: f32 / f16 / int8 per-row / trit per-row / trit g64 (2-bit pack, u32 LUT take)
-- resident bytes, decode step time at B=1/8, logit fidelity vs f32 (02:55 11/17). numpy only, offline."""
import numpy as np, time, json, sys, os
from concurrent.futures import ThreadPoolExecutor
V = int(os.environ.get("V", 32000)); H = 2560; G = 64; SEED = 202611170255
rng = np.random.default_rng(SEED)
# synthetic head: Gaussian rows with per-row norm spread (lognormal) and a few hot input channels
W = rng.standard_normal((V, H), dtype=np.float32) * 0.02
W *= np.exp(rng.normal(0, 0.25, (V, 1))).astype(np.float32)
hot = rng.choice(H, 16, replace=False); W[:, hot] *= 4.0
# --- formats
W16 = W.astype(np.float16)
s8 = (np.abs(W).max(1, keepdims=True) / 127).astype(np.float32); W8 = np.clip(np.rint(W / s8), -127, 127).astype(np.int8)
def trit(Wm, scale):
    return np.clip(np.rint(Wm / scale), -1, 1).astype(np.int8)
gr = np.abs(W).mean(1, keepdims=True).astype(np.float32); T_r = trit(W, gr)
Wg = W.reshape(V, H // G, G); gg = np.abs(Wg).mean(2, keepdims=True).astype(np.float32)
gg16 = gg.astype(np.float16); T_g = trit(Wg, gg).reshape(V, H)
def pack(T):  # 4 trits per byte, codes 0,1,2(-1)
    c = np.where(T < 0, 2, T).astype(np.uint8).reshape(T.shape[0], -1, 4)
    return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).astype(np.uint8)
P_r = pack(T_r); P_g = pack(T_g)
codes = np.arange(256, dtype=np.uint8)
lut = np.stack([(codes >> (2 * k)) & 3 for k in range(4)], 1).astype(np.int8); lut[lut == 2] = -1
LUT_U32 = np.ascontiguousarray(lut).view(np.uint32).ravel()
assert np.array_equal(LUT_U32[P_r].view(np.int8).reshape(V, H), T_r)
GS = np.repeat(gg16.astype(np.float32).reshape(V, H // G), G, axis=1)  # only for reference math
BYTES = {"f32": W.nbytes, "f16": W16.nbytes, "int8_row": W8.nbytes + s8.nbytes,
         "trit_row": P_r.nbytes + gr.nbytes, "trit_g64": P_g.nbytes + gg16.nbytes}
# --- reference (dequantized) heads for fidelity
REF = {"f32": W, "f16": W16.astype(np.float32), "int8_row": W8 * s8, "trit_row": T_r * gr,
       "trit_g64": (T_g.reshape(V, H // G, G) * gg16.astype(np.float32)).reshape(V, H)}
# --- kernels: tiled over vocab rows, W workers, private scratch
def make(kind, T, nw):
    pool = ThreadPoolExecutor(nw) if nw > 1 else None
    scr = {}
    def tile(x, y, o0, n, wid):
        if kind == "f32":
            np.matmul(x, W[o0:o0+n].T, out=y[:, o0:o0+n]); return
        s = scr.setdefault((wid, "f"), np.empty((T, H), np.float32))[:n]
        if kind == "f16":
            np.copyto(s, W16[o0:o0+n])
        elif kind == "int8_row":
            np.copyto(s, W8[o0:o0+n], casting="unsafe")
        else:
            u = scr.setdefault((wid, "u"), np.empty((T, H // 4), np.uint32))[:n]
            np.take(LUT_U32, (P_r if kind == "trit_row" else P_g)[o0:o0+n], out=u)
            np.copyto(s, u.view(np.int8), casting="unsafe")
            if kind == "trit_g64":
                sv = s.reshape(n, H // G, G); sv *= gg16[o0:o0+n].astype(np.float32)
        np.matmul(x, s.T, out=y[:, o0:o0+n])
        if kind == "int8_row": y[:, o0:o0+n] *= s8[o0:o0+n, 0]
        elif kind == "trit_row": y[:, o0:o0+n] *= gr[o0:o0+n, 0]
    def mm(x):
        y = np.empty((x.shape[0], V), np.float32)
        if kind == "f32" and pool is None:
            np.matmul(x, W.T, out=y); return y
        starts = list(range(0, V, T))
        if pool is None:
            for o0 in starts: tile(x, y, o0, min(T, V - o0), 0)
        else:
            chunks = [starts[w::nw] for w in range(nw)]
            def run(w):
                for o0 in chunks[w]: tile(x, y, o0, min(T, V - o0), w)
            list(pool.map(run, range(nw)))
        return y
    return mm
# --- fidelity: hidden states with a planted target at z-score margins
def states(n, z):
    h = rng.standard_normal((n, H), dtype=np.float32)
    t = rng.integers(0, V, n)
    base = h @ W.T; sd = base.std(1)
    w = W[t]; nn = (w * w).sum(1)
    h += ((z * sd)[:, None] * w / nn[:, None]).astype(np.float32)
    return h
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def fidelity():
    out = {}
    for z in (3.0, 5.0, 8.0):
        h = states(256, z); l0 = h @ W.T; p0 = softmax(l0); top0 = np.argsort(-l0, 1)[:, :5]
        for k in ["f16", "int8_row", "trit_row", "trit_g64"]:
            l = h @ REF[k].T; p = softmax(l); top = np.argsort(-l, 1)[:, :5]
            cos = (l * l0).sum(1) / np.linalg.norm(l, axis=1) / np.linalg.norm(l0, axis=1)
            kl = (p0 * (np.log(p0 + 1e-30) - np.log(p + 1e-30))).sum(1)
            out[f"z{z:g}_{k}"] = dict(top1=float((top[:, 0] == top0[:, 0]).mean()),
                top5=float(np.mean([len(set(a) & set(b)) / 5 for a, b in zip(top, top0)])),
                cos=float(np.median(cos)), kl=float(np.median(kl)), p_top0=float(np.median(p0.max(1))))
    return out
def timeit(f, x, rep=7, warm=2):
    for _ in range(warm): f(x)
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter(); f(x); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))
def main(path):
    # check kernels vs reference math
    xc = rng.standard_normal((4, H), dtype=np.float32)
    errs = {k: float(np.abs(make(k, 512, 1)(xc) - xc @ REF[k].T).max()) for k in BYTES}
    fid = fidelity()
    cfgs = [("f32", 0, 1)] + [(k, T, nw) for k in ["f32", "f16", "int8_row", "trit_row", "trit_g64"] for T, nw in ((512, 1), (256, 8))]
    fns = {c: make(*c) for c in cfgs}
    tim = {}
    for B in (1, 8):
        x = rng.standard_normal((B, H), dtype=np.float32)
        rounds = {c: [] for c in cfgs}
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

| format | bytes | MB | int8_row / x | f16 / x |
|---|---:|---:|---:|---:|
| `f32` | 327,680,000 | 327.7 | 0.25x | 0.50x |
| `f16` | 163,840,000 | 163.8 | 0.50x | 1.00x |
| `int8_row` | 82,048,000 | 82.0 | 1.00x | 2.00x |
| `trit_row` | 20,608,000 | 20.6 | 3.98x | 7.95x |
| `trit_g64` | 23,040,000 | 23.0 | 3.56x | 7.11x |

### Next-token fidelity vs f32 (256 states per margin; deterministic, identical in both runs)

| z | format | top-1 | top-5 overlap | logit cos | KL |
|---|---|---:|---:|---:|---:|
| 3 | `f16` | 1.000 | 1.000 | 1.00000 | 2.37e-08 |
| 3 | `int8_row` | 0.938 | 0.973 | 0.99983 | 2.70e-04 |
| 3 | `trit_row` | 0.316 | 0.359 | 0.86612 | 2.63e-01 |
| 3 | `trit_g64` | 0.312 | 0.365 | 0.86883 | 2.57e-01 |
| 5 | `f16` | 1.000 | 1.000 | 1.00000 | 4.10e-08 |
| 5 | `int8_row` | 0.980 | 0.974 | 0.99983 | 2.74e-04 |
| 5 | `trit_row` | 0.293 | 0.416 | 0.86238 | 2.83e-01 |
| 5 | `trit_g64` | 0.312 | 0.423 | 0.86490 | 2.79e-01 |
| 8 | `f16` | 1.000 | 1.000 | 1.00000 | 4.99e-08 |
| 8 | `int8_row` | 1.000 | 0.976 | 0.99983 | 2.66e-04 |
| 8 | `trit_row` | 0.863 | 0.485 | 0.85912 | 5.35e-01 |
| 8 | `trit_g64` | 0.879 | 0.495 | 0.86201 | 5.32e-01 |

### Step time, ms (median of rounds, min–max)

| B | config | run1 | run2 | int8_row (same threading) / x |
|---|---|---:|---:|---:|
| 1 | `f32_T0_W1` | 20.86 (20.42–22.35) | 21.58 (19.77–22.34) | 1.36 / 1.26 |
| 1 | `f32_T512_W1` | 20.59 (19.22–21.35) | 21.04 (19.16–23.20) | 1.38 / 1.29 |
| 1 | `f32_T256_W8` | 22.45 (21.09–24.64) | 22.21 (20.97–24.28) | 0.75 / 0.66 |
| 1 | `f16_T512_W1` | 97.87 (95.94–110.19) | 98.18 (94.95–103.62) | 0.29 / 0.28 |
| 1 | `f16_T256_W8` | 16.25 (14.72–24.01) | 19.01 (16.65–25.68) | 1.04 / 0.77 |
| 1 | `int8_row_T512_W1` | 28.38 (24.70–28.95) | 27.17 (25.06–28.19) | 1.00 / 1.00 |
| 1 | `int8_row_T256_W8` | 16.90 (15.78–17.63) | 14.69 (14.00–15.90) | 1.00 / 1.00 |
| 1 | `trit_row_T512_W1` | 53.60 (51.70–56.30) | 54.00 (52.98–55.78) | 0.53 / 0.50 |
| 1 | `trit_row_T256_W8` | 23.25 (21.75–24.64) | 21.99 (21.63–22.76) | 0.73 / 0.67 |
| 1 | `trit_g64_T512_W1` | 76.77 (74.91–81.48) | 78.50 (77.31–82.72) | 0.37 / 0.35 |
| 1 | `trit_g64_T256_W8` | 23.91 (23.54–23.96) | 24.17 (23.05–26.55) | 0.71 / 0.61 |
| 8 | `f32_T0_W1` | 40.42 (40.11–43.03) | 40.54 (37.51–49.17) | 1.22 / 1.20 |
| 8 | `f32_T512_W1` | 40.77 (39.84–43.56) | 40.40 (39.07–43.06) | 1.21 / 1.20 |
| 8 | `f32_T256_W8` | 5.83 (5.59–8.30) | 8.42 (5.32–10.97) | 1.41 / 1.23 |
| 8 | `f16_T512_W1` | 119.78 (114.83–124.15) | 118.34 (114.55–121.09) | 0.41 / 0.41 |
| 8 | `f16_T256_W8` | 20.34 (18.33–30.52) | 28.12 (18.46–30.48) | 0.40 / 0.37 |
| 8 | `int8_row_T512_W1` | 49.31 (47.68–50.85) | 48.64 (45.83–59.39) | 1.00 / 1.00 |
| 8 | `int8_row_T256_W8` | 8.20 (8.03–11.92) | 10.32 (7.38–13.10) | 1.00 / 1.00 |
| 8 | `trit_row_T512_W1` | 71.09 (69.52–75.12) | 76.17 (73.27–80.23) | 0.69 / 0.64 |
| 8 | `trit_row_T256_W8` | 15.00 (12.88–17.71) | 14.39 (12.35–17.80) | 0.55 / 0.72 |
| 8 | `trit_g64_T512_W1` | 96.72 (95.25–98.08) | 98.37 (96.74–106.18) | 0.51 / 0.49 |
| 8 | `trit_g64_T256_W8` | 16.70 (15.60–21.77) | 16.62 (15.09–20.36) | 0.49 / 0.62 |

## Verdict

**not-viable.** Memory clears the bar easily (trit_row 3.98x smaller than int8 per-row, 7.95x than f16; trit_g64 3.56x / 7.11x), but the error does not: every vocab logit is a 2560-term dot product, so trit rounding noise (cos 0.86) is comparable to the gap between the top candidates and flips the argmax for ~70% of tokens at realistic margins (z 3–5), and still ~13% at a confident z=8. Per-group scales are within noise of per-row. int8 per-row is near-lossless here (cos 0.99983, top-1 0.94–1.00, KL ~3e-4) at 2.0x less than f16. On speed the ternary head is also behind: threaded it runs 0.67–0.73x of int8_row at B=1 and 0.55–0.72x at B=8 (the LUT take costs more than an int8→f32 cast), and f16 single-thread is the slowest path (numpy f16 cast). Keep the LM head at int8 per-row; the ternary win in this stack stays on the hidden-layer weights.

**Caveat:** synthetic Gaussian head and planted-margin states (real logit margins and hot-channel structure differ); V=32000; numpy on 8 cores with box load (~13 GB of 15 GB used by other processes); threaded B=1 timings are noisy.

## Pair files this wave

At Quant Chief 02:05 MT (`last_run` 09:08Z Nov 17), pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave.
- Roadmap blobs unchanged (JuniorLLM docs/BETA_TO_OS.md bac6923, JuniorHome docs/ROADMAP.md 4efb666).
- JuniorLLM head before this push was 0df9b80 (the 18:55 note, ~8 h ago); no other commit in the last 6 hours.
- Chief's LLC skip path (JuniorHome 8459058, i2sd re-advance) is in another repo; no Trit Pack, Handshake or LLC commit touched `junior_osai/vault/findings/` in the last 6 hours.
- This note adds one file under `junior_osai/vault/findings/`.

## Next question

Between int8 per-row (lossless here) and ternary (broken), is there a 1.5x memory step on the LM head: does a **4-bit g64 LM head** (nibble pack, f16 group scales, 4.25 bits) keep top-1 agreement ≥ 0.99 and KL ≤ 1e-3 vs f32 at z 3–8 while holding ≥ 1.5x less than int8 per-row, and what does the nibble expand cost per step at B=1/8 in numpy?

close: sha256(prev || this)[:16] = aaa7a1bd5f3e8c70
