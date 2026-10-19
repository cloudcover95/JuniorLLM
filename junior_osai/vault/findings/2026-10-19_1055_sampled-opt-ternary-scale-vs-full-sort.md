---
local: true
date: 2026-10-19
wave: "2026-10-19 10:05"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 4: JuniorClimbs scripts/stonefield_stdlib_probe.py (no FastAPI required)"
chief_assignment: "pair_A Live beta remaining gaps, prefer steps 4 and 5"
verdict: viable
tags: [ternary, osai, findings, live-beta, step-4, winsor, optimal-ternary, i2s, numpy]
---

# Sampled optimal ternary scale vs full-sort exact optimum vs Winsor gammas

Previous note: [[2026-10-19_0555_pack-only-trit-census-i2s-vs-fp16-fp32]]

## Question
Winsor's step 4 file (JuniorQuant `src/live_beta_step4_stonefield_ternary_gap.py`) measures how far each Winsor gamma's trits are from the exact best ternary code, using the sorted-prefix rule SSE_opt = E - max_k P_k^2 / k. On the 6 to 9 entry StoneField seed that is free. At 2B4T tensor sizes it needs a full sort plus float64 prefix scores. Can a cheaper search (sort a 1-in-61 strided sample to pick the threshold, then one exact pass over the tensor for count and sum) land on the optimum within 0.01% SSE while being at least 1.5x faster or smaller than the full-sort exact method? Side question: how big is the optimum gap of the absmean and clip95 gammas at that scale?

## Method
- numpy only, python3, offline, no sockets, no files read, no installs. Synthetic float32 weights, three 2B4T shapes (2560 x 2560, 2560 x 6912, 6912 x 2560), three kinds: iid Gaussian (sigma 0.02), Student-t df 4 (heavy tail), and a per-row lognormal scale spread (structured, not iid). Two seeds (1055, 1955), 7 timed repeats each. Median, min and max are reported.
- Methods, each returning a threshold and a scale. SSE is scored afterwards in float64, outside the timer:
  - `exact_sort`: full `np.sort` of |W|, float64 cumsum, argmax of P_k^2 / k. This is the reference optimum, the same rule as Winsor's file.
  - `sample61_1pass`: sort |W|[::61] (prime stride, so it never aliases the 2560 or 6912 column counts), take the midpoint threshold at the sample optimum, then one pass over the full tensor: mask, count, float32 dot for the sum, scale = mean of the support.
  - `sample61_lloyd`: the same seed, then alternate threshold = alpha / 2 and alpha = support mean until the support is stable. `lloyd_cold` starts from absmean.
  - `absmean` (gamma = mean|W|) and `clip95` (gamma = mean min(|W|, tau), tau = sorted|W|[floor(0.95 (n - 1))], eps 1e-7). Trits are round-half-even, clipped to -1..1, and gamma is the stored scale.
- Peak extra memory comes from `tracemalloc` on one untimed call (numpy reports its buffers to it).
- Acceptable error: SSE excess over the exact optimum of 0.01% or less, far inside RATE 0.05.

Command (run in a scratch dir, not the repo): `python3 bench_opt_ternary_scale_v2.py 1055` and `python3 bench_opt_ternary_scale_v2.py 1955` (numpy 2.5.4, python 3.13.5, 8 CPUs).

```python
# Live beta step 4 bench: exact optimal ternary code vs O(n) searches vs Winsor gammas.
# numpy only, offline, no sockets, no files read. python3 bench_opt_ternary_scale_v2.py
import sys, time, tracemalloc
import numpy as np

REPS, STRIDE = 7, 61          # prime stride: never aliases 2560 / 6912 column counts

def ev(a64, E, thr, scale):
    """untimed float64 SSE of ||W - scale*t||^2, support |w| > thr (or >= for exact)."""
    m = a64 > thr
    k = int(m.sum()); S = float(a64[m].sum())
    return E - 2 * scale * S + k * scale * scale, E - S * S / k, k

def exact_ref(a):                       # full sort + float64 prefix scores (Winsor file's method)
    s = np.sort(a)[::-1]
    P = np.cumsum(s, dtype=np.float64)
    k = np.arange(1, s.size + 1, dtype=np.float64)
    score = P * P / k
    j = int(np.argmax(score))
    return float(np.nextafter(s[j], np.float32(0))), P[j] / (j + 1)

def opt_on_sorted_desc(s):
    P = np.cumsum(s, dtype=np.float64)
    sc = P * P
    sc /= np.arange(1, s.size + 1, dtype=np.float64)
    j = int(np.argmax(sc))
    return j, P

def sample_once(a):                     # sort a 1/61 strided sample, one exact pass
    s = np.sort(a[::STRIDE])[::-1]
    j, P = opt_on_sorted_desc(s)
    thr = np.float32((s[j] + s[j + 1]) / 2)
    m = a > thr
    k = np.count_nonzero(m)
    S = float(np.dot(a, m.astype(np.float32)))
    return float(thr), S / k

def sample_lloyd(a, iters=20):          # sample seed, then Lloyd to its fixed point (exact KKT point)
    s = np.sort(a[::STRIDE])[::-1]
    j, P = opt_on_sorted_desc(s)
    alpha = P[j] / (j + 1)
    kprev = -1
    for it in range(iters):
        m = a > np.float32(alpha / 2)
        k = np.count_nonzero(m)
        alpha = float(np.dot(a, m.astype(np.float32))) / k
        if k == kprev:
            break
        kprev = k
    sample_lloyd.iters = it + 1
    return float(np.float32(alpha / 2)), alpha

def lloyd_cold(a, iters=50):            # Lloyd from absmean, no sample
    alpha = float(a.mean(dtype=np.float64)); kprev = -1
    for it in range(iters):
        m = a > np.float32(alpha / 2)
        k = np.count_nonzero(m)
        alpha = float(np.dot(a, m.astype(np.float32))) / k
        if k == kprev:
            break
        kprev = k
    lloyd_cold.iters = it + 1
    return float(np.float32(alpha / 2)), alpha

def absmean(a):                         # BitNet b1.58 gamma: quantize is a single mean
    g = float(a.mean(dtype=np.float64)); return g / 2, g

def clip95(a):                          # Winsor clip gamma, tau = sorted|W|[floor(.95(n-1))], eps 1e-7
    i = int(np.floor(0.95 * (a.size - 1)))
    tau = np.partition(a, i)[i]
    g = float(np.minimum(a, tau).mean(dtype=np.float64)); return (g + 1e-7) / 2, g

def timed(fn, a):
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); r = fn(a); ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1e3
    return r, float(np.median(ts)), float(ts.min()), float(ts.max())

def peak(fn, a):
    tracemalloc.start(); fn(a); p = tracemalloc.get_traced_memory()[1]; tracemalloc.stop(); return p

METHODS = [("exact_sort", exact_ref), ("sample61_1pass", sample_once), ("sample61_lloyd", sample_lloyd),
           ("lloyd_cold", lloyd_cold), ("absmean", absmean), ("clip95", clip95)]

def main(seed):
    print(f"# numpy {np.__version__} python {sys.version.split()[0]} seed {seed} reps {REPS}")
    for (r, c) in [(2560, 2560), (2560, 6912), (6912, 2560)]:
        for dist in ("gauss", "t4", "rowscale"):
            rng = np.random.default_rng(seed)
            if dist == "gauss":
                w = rng.standard_normal((r, c), dtype=np.float32) * np.float32(0.02)
            elif dist == "t4":
                w = (rng.standard_t(4, size=(r, c)) * 0.02).astype(np.float32)
            else:                        # per-row scale spread (structured, not iid)
                w = rng.standard_normal((r, c), dtype=np.float32) * rng.lognormal(-4, 0.5, (r, 1)).astype(np.float32)
            a = np.abs(w).ravel(); a64 = a.astype(np.float64); E = float(a64 @ a64)
            res = {}
            for name, fn in METHODS:
                (thr, scale), t, t0, t1 = timed(fn, a)
                sse, sse_k, k = ev(a64, E, thr, scale)
                res[name] = (t, t0, t1, peak(fn, a), sse, sse_k, k, getattr(fn, "iters", ""))
            opt = res["exact_sort"][4]; base = res["exact_sort"][0]
            print(f"## {r}x{c} {dist} n={a.size} fp32_bytes={4*a.size} i2s_bytes={a.size//4} rel_l2_opt={(opt/E)**.5:.4f}")
            for name, v in res.items():
                print(f"  {name:15s} med={v[0]:8.2f}ms ({v[1]:.2f}-{v[2]:.2f}) peak={v[3]/1e6:7.1f}MB "
                      f"excess={100*(v[4]/opt-1):9.5f}% support_only={100*(v[5]/opt-1):8.5f}% k={v[6]} it={v[7]} "
                      f"vs_exact={base/v[0]:.2f}x")
            del w, a, a64

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1055)
```

## Results
Seed 1055, with seed 1955 in brackets. SSE excess = SSE / SSE_opt - 1. The I2_S pack is n / 4 bytes, 16x smaller than fp32, on every row. The Gaussian and t4 rows for 2560 x 6912 and 6912 x 2560 use the same draws reshaped, so their excess numbers match. The rowscale rows differ.

| shape | weights | exact sort ms, median (min-max) [seed 1955] | sampled 1-pass ms [seed 1955] | speedup | peak MB exact / sampled | sampled SSE excess vs optimum | sample+Lloyd ms (iters) | absmean SSE excess | clip95 SSE excess |
|---|---|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 107.00 (92.74-200.97) [117.53] | 4.97 (4.85-9.75) [4.97] | 21.5x [23.6x] | 183.5 / 34.1 | 0.00024% [0.00004%] | 42.83 (10) | 38.45112% | 42.25482% |
| 2560x2560 | t4 | 97.45 (95.58-134.54) [88.10] | 6.17 (5.34-10.62) [5.38] | 15.8x [16.4x] | 183.5 / 34.1 | 0.00290% [0.00071%] | 74.06 (17) | 30.29606% | 35.82189% |
| 2560x2560 | rowscale | 98.73 (88.55-129.47) [95.84] | 5.44 (5.11-11.70) [4.93] | 18.1x [19.4x] | 183.5 / 34.1 | 0.00330% [0.00007%] | 66.91 (16) | 39.30085% | 45.14329% |
| 2560x6912 | gauss | 301.27 (275.82-506.42) [311.04] | 20.60 (18.86-33.38) [24.91] | 14.6x [12.5x] | 495.5 / 92.0 | 0.00071% [0.00021%] | 211.00 (12) | 38.45307% | 42.25707% |
| 2560x6912 | t4 | 277.86 (271.94-446.91) [279.82] | 23.75 (21.26-31.47) [18.73] | 11.7x [14.9x] | 495.5 / 92.0 | 0.00343% [0.00061%] | 287.95 (18) | 30.08587% | 35.58541% |
| 2560x6912 | rowscale | 277.72 (272.96-440.99) [282.17] | 18.78 (16.63-31.50) [25.62] | 14.8x [11.0x] | 495.5 / 92.0 | 0.00004% [0.00110%] | 301.01 (18) | 37.90577% | 43.83643% |
| 6912x2560 | gauss | 277.19 (274.59-459.10) [287.21] | 24.20 (18.39-32.92) [17.48] | 11.5x [16.4x] | 495.5 / 92.0 | 0.00071% [0.00021%] | 190.46 (12) | 38.45307% | 42.25707% |
| 6912x2560 | t4 | 275.94 (266.57-477.23) [272.52] | 20.87 (18.30-31.63) [20.39] | 13.2x [13.4x] | 495.5 / 92.0 | 0.00343% [0.00061%] | 277.20 (18) | 30.08587% | 35.58541% |
| 6912x2560 | rowscale | 283.13 (267.71-462.49) [290.58] | 22.10 (19.52-32.21) [18.16] | 12.8x [16.0x] | 495.5 / 92.0 | 0.00001% [0.00055%] | 203.15 (13) | 38.78065% | 44.68131% |

- Both Lloyd variants reach the optimum (excess 0.00000%). The cold start needs 15 to 28 full passes and is 0.55x to 1.69x the speed of exact sort. The sample-seeded start needs 9 to 18 passes and is 0.82x to 3.23x. Lloyd is not a reliable win. The single sampled pass is.
- The sampled support can differ from the exact one by up to 0.8% of entries (for example 7,389,227 vs 7,329,637 on t4), but the SSE surface is flat at the optimum, so the excess stays at 0.0033% or less across all 18 cases.
- absmean's trits sit 30% to 40% above the optimum SSE. Storing the support mean in place of gamma, with the identical pack, still leaves 12% to 17% excess. So about two thirds of absmean's gap is the scale and one third is the support. clip95 is worse on all 18 cases, at 35.6% to 45.7%. The optimal rel_l2 is 0.436 (Gaussian) to 0.561 (t4).
- absmean costs 2.5 to 7.3 ms, 2x to 3x cheaper than the sampled pass, but at 30% to 40% more error.

## Verdict
**Viable.** The sampled one-pass search reaches the exact ternary optimum within 0.0033% SSE (worst of 18 cases) while running 11.0x to 23.6x faster than the full-sort exact method (median 5 to 26 ms vs 88 to 311 ms). Its peak extra memory is 5.4x smaller (92 MB vs 496 MB on the 17.7M-entry FFN tensors, and 34 MB vs 184 MB on 2560 x 2560). So step 4's optimum-gap check can run at model scale, not just on the seed. The same measurement shows that Winsor's absmean and clip95 gammas leave 30% to 46% SSE on the table against a free optimum. Caveats: synthetic weights; SSE is per-tensor weight error, not task accuracy; a strided sample assumes no structure at period 61, which the rowscale case supports but does not prove; the exact method's float64 prefix arrays dominate its memory, and a chunked exact pass could narrow that gap. On the 6 to 9 entry StoneField seed, keep the exact Fraction method. This finding applies only once the check runs on real I2_S-sized packs.

## This wave's pair files
- Pair A (Live beta remaining gaps, prefer steps 4 and 5): Winsor JuniorQuant 4e55433 [`src/live_beta_step4_stonefield_ternary_gap.py`](https://github.com/cloudcover95/JuniorQuant/blob/4e55433/src/live_beta_step4_stonefield_ternary_gap.py) (the item this note benchmarks); Trit Pack JuniorLLM 3ca0657 [`scripts/asahi_mlx_cluster_shard_i2s_key.py`](https://github.com/cloudcover95/JuniorLLM/blob/3ca0657/scripts/asahi_mlx_cluster_shard_i2s_key.py) (took the ROADMAP #4 Asahi MLX fallback; its header says the visible Chief assignment was stale at 10:20).
- Pair B (JuniorOS overlay remaining gaps): Handshake JuniorLLM 4a2b958 [`ports/overlay_i2sd_install_closure_handshake.py`](https://github.com/cloudcover95/JuniorLLM/blob/4a2b958/ports/overlay_i2sd_install_closure_handshake.py); Side-Commit stocksnode 4c83ae0 [`src/stocksnode/core/juniorctl_health_output_receipt_ledger.py`](https://github.com/cloudcover95/stocksnode/blob/4c83ae0/src/stocksnode/core/juniorctl_health_output_receipt_ledger.py) (filed under Live beta step 1 health, not the overlay item).

## Next question
Does storing the optimal scale per 128-trit I2_S block (one fp16 per block, about 6% more bytes) cut SSE by 1.5x or more over the per-tensor optimum at the same speed, or is per-tensor optimal already the knee? For step 5, can `backfill_check` exit-code runs be gated with the sampled pass instead of a full re-derive?

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-19_0555_pack-only-trit-census-i2s-vs-fp16-fp32.md.


close: b717c77c808d9b81
