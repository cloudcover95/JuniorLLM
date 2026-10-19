---
local: true
date: 2026-10-19
wave: "2026-10-19 14:05"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: I2_S 2B4T only if GGUF already on disk (scale granularity of the 128-trit I2_S block pack bitnetd would serve)"
chief_assignment: "pair_A JuniorOS overlay remaining gaps"
verdict: not-viable
tags: [ternary, osai, findings, junioros-overlay, i2s, block-scale, optimal-ternary, numpy]
---

# Per-block optimal ternary scale (fp16 per 128-trit I2_S block) vs per-tensor optimum

Previous note: [[2026-10-19_1055_sampled-opt-ternary-scale-vs-full-sort]]

## Question
The previous note found the exact per-tensor ternary optimum is cheap to reach at 2B4T sizes. It asked next whether one optimal fp16 scale per 128-trit I2_S block (2 bytes per 32 bytes of codes, so 6.25% more bytes) cuts weight SSE by 1.5x or more over that per-tensor optimum at the same speed, or whether per-tensor is already the knee. This wave's Trit Pack file keys the bitnetd overlay trits into an I2_S 128-block pack with one per-tensor gamma, so the answer decides whether that pack format should grow a per-block scale table.

Viability bar: SSE x bytes at least 1.5x better than the per-tensor optimum (so the 6.25% byte cost counts against the gain), with packed matvec no more than about 1.1x slower.

## Method
- python3 + numpy 2.5.4 only, offline, no sockets, no files read, no installs, 8 cores. Synthetic float32 weights, two 2B4T shapes (2560 x 2560 and 2560 x 6912), two seeds (1455, 2455), four weight kinds:
  - `gauss`: iid Gaussian, sigma 0.02.
  - `t4`: Student-t df 4 (heavy tail).
  - `rowscale`: Gaussian times a per-row lognormal scale (sigma 0.5), structure along output rows.
  - `blockscale`: Gaussian times a lognormal scale per (row, 128-column block), structure at exactly the I2_S block size.
- Every scheme uses the exact optimum for its group, the same sorted-prefix rule as the previous note: sort |w| descending, k* = argmax P_k^2 / k, keep sign(w) for the top k*, scale = P_k* / k*. Groups: whole tensor, each row, each block of 64, 128 or 256. A per-block absmean code (gamma = mean |w|, round, clip) is the cheap reference. Scales are stored as fp16; SSE is scored in float64 outside the timer.
- Bytes: 2 bits per weight plus 2 bytes per scale.
- Quantize timing: 5 repeats per case. Packed matvec timing: 9 repeats, from 2-bit packed bytes through the same 256 x 4 float32 LUT decode, per-tensor (`(W @ x) * s`) vs per-block (`einsum` over 128-wide block partials, then scale and sum; checked equal to the scale-folded decode). Output error is rel_l2 of W_hat @ x against float64 W @ x for one random x.
- Medians with min-max over both seeds. SSE columns are the mean of the two seeds (they agree to 3 to 4 digits).

Command (run from a scratch dir on the box):

```
OMP_NUM_THREADS=8 python3 block_scale_bench.py > results.json
```

## Results
Weight error (SSE / ||W||^2), lower is better. Gain columns are per-tensor optimum SSE divided by the scheme's SSE, over both seeds. "SSE x bytes gain" divides that by the byte ratio.

| shape | weights | SSE/E tensor opt | row opt | block128 opt | block64 opt | block256 opt | block128 absmean | tensor/block128 | tensor/row | block128 SSE x bytes gain | row SSE x bytes gain |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 0.1901 | 0.1899 | 0.1870 | 0.1838 | 0.1885 | 0.2611 | 1.017-1.017x | 1.001-1.001x | 0.957-0.957x | 0.998-0.998x |
| 2560x2560 | t4 | 0.3131 | 0.3122 | 0.2914 | 0.2719 | 0.3023 | 0.4038 | 1.073-1.076x | 1.003-1.003x | 1.010-1.013x | 1.000-1.000x |
| 2560x2560 | rowscale | 0.2944 | 0.1900 | 0.1871 | 0.1839 | 0.1887 | 0.2612 | 1.572-1.574x | 1.548-1.550x | 1.480-1.482x | 1.543-1.545x |
| 2560x2560 | blockscale | 0.2935 | 0.2816 | 0.1868 | 0.1837 | 0.2172 | 0.2610 | 1.561-1.581x | 1.039-1.046x | 1.469-1.488x | 1.036-1.042x |
| 2560x6912 | gauss | 0.1902 | 0.1902 | 0.1871 | 0.1839 | 0.1886 | 0.2611 | 1.017-1.017x | 1.000-1.000x | 0.957-0.957x | 0.999-0.999x |
| 2560x6912 | t4 | 0.3131 | 0.3129 | 0.2911 | 0.2718 | 0.3020 | 0.4038 | 1.073-1.078x | 1.001-1.001x | 1.010-1.014x | 0.999-1.000x |
| 2560x6912 | rowscale | 0.2944 | 0.1902 | 0.1871 | 0.1840 | 0.1887 | 0.2612 | 1.561-1.586x | 1.536-1.560x | 1.469-1.493x | 1.534-1.558x |
| 2560x6912 | blockscale | 0.2926 | 0.2887 | 0.1870 | 0.1839 | 0.2172 | 0.2611 | 1.563-1.566x | 1.013-1.014x | 1.471-1.474x | 1.012-1.012x |

Bytes, quantize time and packed matvec time (ms, median (min-max) over both seeds):

| shape | weights | bytes tensor / row / block128 / fp16 | quant ms tensor opt | quant ms block128 opt | quant ms row opt | matvec ms fp32 | packed matvec ms per-tensor | packed matvec ms block128 (einsum) | block128 / tensor matvec | y rel_l2 tensor / block128 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 1,638,402 / 1,643,520 / 1,740,800 / 13,107,200 | 119.0 (107.6-221.2) | 87.0 (82.1-94.5) | 97.2 (93.2-101.4) | 0.2 (0.1-0.3) | 16.4 (14.8-17.9) | 17.5 (16.4-19.3) | 1.07x | 0.432 / 0.426 |
| 2560x2560 | t4 | 1,638,402 / 1,643,520 / 1,740,800 / 13,107,200 | 109.0 (105.2-166.9) | 88.1 (79.3-96.6) | 89.6 (80.4-96.5) | 0.1 (0.1-2.4) | 15.6 (14.3-21.1) | 16.7 (16.0-20.6) | 1.07x | 0.563 / 0.545 |
| 2560x2560 | rowscale | 1,638,402 / 1,643,520 / 1,740,800 / 13,107,200 | 121.3 (113.3-161.3) | 94.2 (90.9-99.3) | 96.2 (89.7-106.7) | 0.1 (0.1-0.3) | 18.6 (15.3-33.4) | 17.5 (16.0-19.9) | 0.94x | 0.550 / 0.451 |
| 2560x2560 | blockscale | 1,638,402 / 1,643,520 / 1,740,800 / 13,107,200 | 123.9 (114.8-153.4) | 91.5 (85.8-105.5) | 92.3 (85.4-107.4) | 0.1 (0.1-0.3) | 19.0 (14.6-24.7) | 17.0 (16.1-19.3) | 0.90x | 0.550 / 0.434 |
| 2560x6912 | gauss | 4,423,682 / 4,428,800 / 4,700,160 / 35,389,440 | 354.8 (345.7-601.5) | 286.2 (280.1-358.8) | 292.7 (280.3-365.6) | 0.3 (0.3-3.2) | 46.3 (44.0-51.1) | 49.3 (46.8-54.9) | 1.06x | 0.429 / 0.427 |
| 2560x6912 | t4 | 4,423,682 / 4,428,800 / 4,700,160 / 35,389,440 | 358.8 (347.7-477.8) | 293.2 (287.6-354.7) | 296.0 (289.3-374.8) | 0.4 (0.3-1.1) | 51.6 (44.3-57.0) | 52.2 (46.3-59.6) | 1.01x | 0.556 / 0.530 |
| 2560x6912 | rowscale | 4,423,682 / 4,428,800 / 4,700,160 / 35,389,440 | 354.7 (339.5-543.7) | 291.0 (282.6-365.7) | 296.6 (291.5-329.0) | 0.5 (0.3-3.8) | 48.8 (45.0-57.2) | 51.0 (46.7-58.4) | 1.05x | 0.531 / 0.445 |
| 2560x6912 | blockscale | 4,423,682 / 4,428,800 / 4,700,160 / 35,389,440 | 362.0 (347.5-500.1) | 280.6 (278.0-342.5) | 295.6 (287.3-326.4) | 0.3 (0.3-1.9) | 53.0 (45.3-71.4) | 52.4 (47.4-67.1) | 0.99x | 0.537 / 0.423 |

- On iid Gaussian weights, per-block scales buy almost nothing: 1.017x lower SSE, which is 0.957x once the 6.25% byte cost is counted. Heavy-tailed t4 gets 1.07x, or 1.01x after bytes. Per-tensor is the knee for weights with no scale structure.
- Per-block scales clear 1.5x raw SSE only when the weight scale actually varies at or above block granularity (rowscale 1.56 to 1.59x, blockscale 1.56 to 1.58x). Even then, SSE x bytes tops out at 1.47 to 1.49x, just under the bar.
- When the structure is per row, one fp16 per row (5,120 bytes, +0.3%) gets 1.54 to 1.56x, nearly all of what per-block gets, and that does clear 1.5x after bytes. Per-row only fails when the structure sits inside rows (blockscale 1.01 to 1.05x).
- Smaller blocks help a little more: block 64 (+12.5% bytes) reaches 0.184 to 0.272; block 256 is in between. absmean per 128-block is worse than the per-tensor optimum on every kind (0.261 vs 0.190 on Gaussian), so a per-block table needs the optimal rule, not absmean.
- Cost is not the problem. The exact per-block optimum is 0.73x to 0.82x the time of the per-tensor exact sort (many short sorts beat one long sort), and the per-block packed matvec runs at 0.90x to 1.07x of the per-tensor one. Both packed paths are dominated by the numpy LUT decode (15 to 53 ms), as earlier notes found. The fp32 matvec column (0.1 to 0.5 ms) is the BLAS baseline on a warm, preloaded float32 matrix. It is listed for scale only and is not part of the comparison.

## Verdict
**Not viable as a default.** One optimal fp16 scale per 128-trit I2_S block never reached 1.5x better SSE x bytes than the per-tensor optimum in 16 cases (best 1.49x, on synthetic block-structured weights). It gives 0.96x to 1.01x on iid and heavy-tailed weights. Speed is a wash (matvec 0.90x to 1.07x, quantize faster), so the only cost is 6.25% more bytes, but the gain is not there unless the weights carry block-level scale structure. If the structure is per row, a per-row scale gets the same gain for 0.3% more bytes. Keep the I2_S 128-block pack on one per-tensor gamma, as Trit Pack's key does, and use the exact per-tensor optimum rather than absmean. Caveats: synthetic weights only, with no GGUF on disk (the overlay rule forbids pulling one); SSE is weight error, not task accuracy. Real BitNet b1.58 2B4T weights are trained against a per-tensor absmean scale, so they are already ternary and this question only matters for post-training ternarization of float tensors.

## This wave's pair files
- Pair A (JuniorOS overlay remaining gaps): Trit Pack BitNet-mlx 202b305 [`scripts/bitnetd_restart_reach_i2s_key.py`](https://github.com/cloudcover95/BitNet-mlx/blob/202b305/scripts/bitnetd_restart_reach_i2s_key.py) (bitnetd restart, ordering and sandbox-reach trits into JTR1 and an I2_S 128-block key with one per-tensor gamma, the format this note benchmarks); Winsor JuniorStock 67ebb13 [`quant/calibration/slot1900_receipt_age_trit_gate.py`](https://github.com/cloudcover95/JuniorStock/blob/67ebb13/quant/calibration/slot1900_receipt_age_trit_gate.py) (filed under the 19:00 second slice, not the overlay item).
- Pair B (Live beta step 5): Handshake JuniorLLM 829a789 [`ports/live_beta_backfill_coverage_handshake.py`](https://github.com/cloudcover95/JuniorLLM/blob/829a789/ports/live_beta_backfill_coverage_handshake.py); Side-Commit web3node 11ff8e3 [`second_slice_age_gate_ledger.py`](https://github.com/cloudcover95/web3node/blob/11ff8e3/second_slice_age_gate_ledger.py) (filed under ROADMAP Automations 19:00 second slice, not step 5).

## Inline script
`block_scale_bench.py`:

```python
#!/usr/bin/env python3
"""Per-block optimal ternary scale (fp16 per 128-trit I2_S block) vs per-tensor optimum.
numpy only, offline, synthetic weights. Prints a JSON blob of results."""
import json, sys, time
import numpy as np

SHAPES = [(2560, 2560), (2560, 6912)]
KINDS = ["gauss", "t4", "rowscale", "blockscale"]
SEEDS = [1455, 2455]
REP_Q = 5
REP_MV = 9
B = 128

def make(kind, shape, rng):
    r, c = shape
    if kind == "gauss":
        w = rng.standard_normal(shape, dtype=np.float32) * 0.02
    elif kind == "t4":
        w = (rng.standard_t(4, size=shape) * 0.02).astype(np.float32)
    elif kind == "rowscale":
        w = rng.standard_normal(shape, dtype=np.float32) * 0.02
        w *= np.exp(rng.standard_normal((r, 1)) * 0.5).astype(np.float32)
    elif kind == "blockscale":  # scale varies per 128-column block (channel-group outliers)
        w = rng.standard_normal(shape, dtype=np.float32) * 0.02
        w *= np.exp(rng.standard_normal((r, c // B, 1)) * 0.5).astype(np.float32).repeat(B, axis=2).reshape(r, c)
    return np.ascontiguousarray(w)

def opt_rows(a):
    """Exact optimal ternary code for each row of a (m, n): returns trits int8, scale float64 per row."""
    s = np.sort(np.abs(a), axis=1)[:, ::-1].astype(np.float64)
    P = np.cumsum(s, axis=1)
    k = np.arange(1, a.shape[1] + 1, dtype=np.float64)
    score = P * P / k
    kb = np.argmax(score, axis=1)  # index -> k = kb+1
    kk = kb + 1
    alpha = P[np.arange(a.shape[0]), kb] / kk
    thr = s[np.arange(a.shape[0]), kb]  # keep |w| >= k-th largest
    t = (np.abs(a) >= thr[:, None].astype(a.dtype)) * np.sign(a)
    return t.astype(np.int8), alpha

def q_tensor(w):
    t, a = opt_rows(w.reshape(1, -1))
    return t.reshape(w.shape), np.float16(a[0])

def q_block(w, bs=B):
    t, a = opt_rows(w.reshape(-1, bs))
    return t.reshape(w.shape), a.astype(np.float16)

def q_row(w):
    t, a = opt_rows(w)
    return t, a.astype(np.float16)

def q_absmean_block(w, bs=B):
    blk = w.reshape(-1, bs)
    g = np.abs(blk).mean(axis=1, dtype=np.float64) + 1e-12
    t = np.clip(np.rint(blk / g[:, None]), -1, 1).astype(np.int8)
    return t.reshape(w.shape), g.astype(np.float16)

def recon(t, scale, mode, shape, bs=B):
    tf = t.astype(np.float32)
    if mode == "tensor":
        return tf * np.float32(scale)
    if mode == "row":
        return tf * scale.astype(np.float32)[:, None]
    return (tf.reshape(-1, bs) * scale.astype(np.float32)[:, None]).reshape(shape)

def pack2(t):
    c = (t.reshape(-1) + 1).astype(np.uint8).reshape(-1, 4)  # 0,1,2 codes
    return (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).astype(np.uint8)

LUT = np.array([[((b >> (2 * i)) & 3) - 1 for i in range(4)] for b in range(256)], dtype=np.float32)

def mv_tensor(packed, scale, x, shape):
    W = LUT[packed].reshape(shape)
    return (W @ x) * np.float32(scale)

def mv_block(packed, scale32, x, shape, bs=B):
    r, c = shape
    W = LUT[packed].reshape(r, c // bs, bs)
    part = np.einsum("rbk,bk->rb", W, x.reshape(c // bs, bs), optimize=True)
    return (part * scale32.reshape(r, c // bs)).sum(axis=1)

def mv_block_fold(packed, scale32, x, shape, bs=B):
    r, c = shape
    W = (LUT[packed].reshape(-1, bs) * scale32[:, None]).reshape(r, c)
    return W @ x

def timeit(f, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(min(ts)), float(max(ts))

def sse(w, wh):
    d = (w.astype(np.float64) - wh.astype(np.float64))
    return float((d * d).sum())

def main():
    out = []
    for shape in SHAPES:
        n = shape[0] * shape[1]
        for kind in KINDS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                w = make(kind, shape, rng)
                E = float((w.astype(np.float64) ** 2).sum())
                x = rng.standard_normal(shape[1], dtype=np.float32)
                y_ref = w.astype(np.float64) @ x.astype(np.float64)
                row = {"shape": f"{shape[0]}x{shape[1]}", "kind": kind, "seed": seed, "n": n}
                res = {}
                for name, fn, mode in [("tensor_opt", q_tensor, "tensor"), ("row_opt", q_row, "row"),
                                       ("block128_opt", q_block, "block"), ("block128_absmean", q_absmean_block, "block"),
                                       ("block64_opt", lambda w: q_block(w, 64), "block64"),
                                       ("block256_opt", lambda w: q_block(w, 256), "block256")]:
                    bs = {"block64": 64, "block256": 256}.get(mode, B)
                    m = "block" if mode.startswith("block") else mode
                    tq = timeit(lambda: fn(w), REP_Q)
                    t, s = fn(w)
                    wh = recon(t, s, m, shape, bs)
                    e = sse(w, wh)
                    nscale = 1 if m == "tensor" else (shape[0] if m == "row" else n // bs)
                    by = n // 4 + 2 * nscale
                    yh = wh.astype(np.float64) @ x.astype(np.float64)
                    res[name] = {"quant_ms": tq, "sse_rel": e / E, "bytes": by,
                                 "y_rel_l2": float(np.linalg.norm(yh - y_ref) / np.linalg.norm(y_ref))}
                    if name in ("tensor_opt", "block128_opt"):
                        res[name]["_t"], res[name]["_s"] = t, s
                # matvec timing from packed bytes
                pt = pack2(res["tensor_opt"]["_t"]); pb = pack2(res["block128_opt"]["_t"])
                st = res["tensor_opt"]["_s"]; sb32 = res["block128_opt"]["_s"].astype(np.float32)
                w32 = w
                yt = mv_tensor(pt, st, x, shape); yb = mv_block(pb, sb32, x, shape); yf = mv_block_fold(pb, sb32, x, shape)
                assert np.allclose(yb, yf, rtol=1e-3, atol=1e-3 * np.abs(yf).max())
                row["mv_fp32_ms"] = timeit(lambda: w32 @ x, REP_MV)
                row["mv_tensor_ms"] = timeit(lambda: mv_tensor(pt, st, x, shape), REP_MV)
                row["mv_block_einsum_ms"] = timeit(lambda: mv_block(pb, sb32, x, shape), REP_MV)
                row["mv_block_fold_ms"] = timeit(lambda: mv_block_fold(pb, sb32, x, shape), REP_MV)
                for k in ("tensor_opt", "block128_opt"):
                    res[k].pop("_t"); res[k].pop("_s")
                row["methods"] = res
                row["fp16_bytes"] = 2 * n
                out.append(row)
                print(json.dumps({"shape": row["shape"], "kind": kind, "seed": seed,
                                  "sse_t": res["tensor_opt"]["sse_rel"], "sse_b": res["block128_opt"]["sse_rel"]}), file=sys.stderr, flush=True)
    print(json.dumps(out))

if __name__ == "__main__":
    main()
```

## Next question
Per-row optimal scale is the only cheap knee here (+0.3% bytes, 1.54x lower SSE on row-structured weights). Do any float weight tensors already on disk in the local repos (no downloads) carry enough row-scale spread for per-row ternary to beat per-tensor by 1.5x SSE x bytes? Failing that, does Live beta step 5's `backfill_check` presence gate gain anything from a 2-bit per-path trit pack over its current list scan?

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-19_1055_sampled-opt-ternary-scale-vs-full-sort.md.


close: 9f18ff37993ad6f5
