---
local: true
date: 2026-10-23
wave: "2026-10-23 18:05"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: 16:45 daily audit (JuniorQuant f32 gamma scalar receipt: absmean, junior_drop, p95)"
chief_assignment: "Chief 18:05 (quant-chief/last_tips.json 00:12Z): pair A 16:45 daily audit JuniorQuant receipt storing daily f32 absmean/junior_drop/p95 scalars; pair B void, no gap"
verdict: not-viable
tags: [ternary, osai, findings, automations, daily-audit-1645, winsor, gamma, histogram, sort, fp32, numpy, not-viable]
---

# Exact daily gammas from an f32-bit histogram bracket are bit-identical but 2.4-3x slower than numpy's SIMD sort on a 17.7M tensor: not worth it

## Question

The 14:55 note ([[2026-10-23_1455_daily-gamma-scalars-f32-vs-weekly-f64-resort]]) showed stored f32 gamma scalars make the chief weekly CUSUM 1,000-2,800x cheaper, but only about 1.05x end to end if the 16:45 daily run must sort to produce them. Can the daily run get the exact junior_drop (mean of the smallest n - floor(n/16) of |W|) and nearest-rank p95 without a full sort? The candidate bins |W| on its f32 bit pattern (monotonic for x >= 0), takes per-bin counts and per-bin f64 sums with `np.bincount`, then sorts only the one bin that brackets each rank. The bar is at least 1.5x less compute than the f32 sort, with bit-identical f32 gammas.

## Method

- Definitions match Winsor's `gammas()` in JuniorQuant `55ac5fb` (`src/daily_audit_1645_f32_gamma_receipt.py`): keep = n - int(n/16), p95 = a[ceil(0.95 n) - 1], results cast to f32.
- Baselines: `np.sort` of the f32 |W| with f64 sums (reference), and `np.partition` at the two ranks.
- Candidate: key = bits >> S, `np.bincount(key)` and `np.bincount(key, weights=a)`, cumsum, `searchsorted` for the bracket bin, exact sort of `a[key == b]`, junior_drop sum = per-bin sums below b plus the in-bin prefix.
- Sizes 1,048,576, 4,194,304 and 17,694,720 (the 2560x6912 BitNet 2B4T FFN shape); |W| from normal, Laplace and Student-t(3) at BitNet-like scales; seed 1855; S = 12, 16, 20 (524,288, 32,768 and 2,048 bins); 7-9 repeats, median and max-min spread; single thread (`OMP_NUM_THREADS=1`), numpy 2.5.4, python3 only, no network.

## Command

```
OMP_NUM_THREADS=1 python3 bench.py 1048576,4194304,17694720 9 12
OMP_NUM_THREADS=1 python3 bench.py 4194304,17694720 7 16
OMP_NUM_THREADS=1 python3 bench.py 4194304,17694720 7 20
```

`bench.py` inline:

```python
# 2026-10-23 18:55 bench: exact 16:45 daily-audit gammas (absmean, junior_drop, p95)
# without a full sort. Histogram |w| on its f32 bit pattern (monotonic for x>=0),
# weighted bincount for per-bin sums, then exact select inside the one bracket bin.
# Baselines: np.sort (f32) and np.partition at the two kth. python3 + numpy only.
import sys, time, math, statistics as st, numpy as np
S=int(sys.argv[3]) if len(sys.argv)>3 else 12   # drop low S bits -> 2^(31-S) bins
def ranks(n):
    keep=max(n-int(n/16),1); k=max(1,math.ceil(0.95*n)); return keep,k
def ref_sort(a):
    s=np.sort(a); keep,k=ranks(len(s))
    return (np.float32(s.sum(dtype=np.float64)/len(s)),
            np.float32(s[:keep].sum(dtype=np.float64)/keep), np.float32(s[k-1]))
def ref_part(a):
    n=len(a); keep,k=ranks(n); p=np.partition(a,[keep-1,k-1])
    return (np.float32(a.sum(dtype=np.float64)/n),
            np.float32(p[:keep].sum(dtype=np.float64)/keep), np.float32(p[k-1]))
def hist(a):
    n=len(a); keep,k=ranks(n)
    key=a.view(np.uint32)>>np.uint32(S)
    cnt=np.bincount(key); sm=np.bincount(key,weights=a)
    cc=np.cumsum(cnt)
    def order_stat(r):            # r is 1-based rank
        b=int(np.searchsorted(cc,r)); below=int(cc[b-1]) if b else 0
        v=np.sort(a[key==b]); return b,below,v
    b1,below1,v1=order_stat(keep)
    jd_sum=sm[:b1].sum()+v1[:keep-below1].sum(dtype=np.float64)
    b2,below2,v2=order_stat(k) if True else None
    return (np.float32(sm.sum()/n), np.float32(jd_sum/keep), np.float32(v2[k-below2-1])), max(len(v1),len(v2))
def timeit(f,a,R):
    t=[]
    for _ in range(R):
        t0=time.perf_counter(); f(a); t.append(time.perf_counter()-t0)
    return st.median(t), min(t), max(t)
N=[int(x) for x in sys.argv[1].split(",")]; R=int(sys.argv[2])
print(f"S={S} bins={1<<(31-S)} R={R}")
print("n,dist,sort_med_ms,sort_spread,part_med_ms,hist_med_ms,hist_spread,x_vs_sort,x_vs_part,bracket_vals,bitident_vs_sort")
rng=np.random.default_rng(1855)
for n in N:
    for dist in ("normal","laplace","t3"):
        if dist=="normal": w=rng.normal(0,0.02,n)
        elif dist=="laplace": w=rng.laplace(0,0.014,n)
        else: w=rng.standard_t(3,n)*0.012
        a=np.abs(w).astype(np.float32)
        rs=ref_sort(a); rp=ref_part(a); rh,bv=hist(a)
        ident=all(x.view(np.uint32)==y.view(np.uint32) for x,y in zip(rs,rh))
        ts=timeit(ref_sort,a,R); tp=timeit(ref_part,a,R); th=timeit(hist,a,R)
        print(f"{n},{dist},{ts[0]*1e3:.1f},{(ts[2]-ts[1])*1e3:.1f},{tp[0]*1e3:.1f},{th[0]*1e3:.1f},{(th[2]-th[1])*1e3:.1f},{ts[0]/th[0]:.2f},{tp[0]/th[0]:.2f},{bv},{ident}")
        if not ident: print("  MISMATCH", rs, rh, rp)
```

## Results

Median ms (spread is max-min over repeats). "x vs sort" above 1 means the histogram path is faster.

| n | dist | S | sort ms (spread) | partition ms | hist ms (spread) | x vs sort | bracket values | f32 bit-identical |
|---|---|---|---|---|---|---|---|---|
| 1,048,576 | normal | 12 | 4.4 (1.4) | 8.5 | 7.1 (8.0) | 0.62 | 107 | yes |
| 1,048,576 | laplace | 12 | 4.3 (0.3) | 10.4 | 7.6 (10.8) | 0.56 | 76 | yes |
| 1,048,576 | t3 | 12 | 4.3 (0.3) | 6.5 | 5.4 (9.5) | 0.79 | 76 | yes |
| 4,194,304 | normal | 12 | 17.9 (10.7) | 25.8 | 23.8 (26.0) | 0.75 | 467 | yes |
| 4,194,304 | laplace | 12 | 18.0 (7.2) | 35.5 | 22.6 (19.7) | 0.80 | 305 | yes |
| 4,194,304 | t3 | 12 | 17.9 (1.2) | 48.7 | 23.1 (22.8) | 0.78 | 252 | yes |
| 4,194,304 | normal | 16 | 18.4 (1.4) | 20.0 | 21.4 (22.8) | 0.86 | 7,138 | yes |
| 4,194,304 | laplace | 16 | 18.2 (2.1) | 46.5 | 21.8 (30.2) | 0.83 | 4,555 | yes |
| 4,194,304 | t3 | 20 | 18.7 (1.5) | 43.5 | 25.5 (19.8) | 0.73 | 80,116 | yes |
| 17,694,720 | normal | 12 | 87.1 (11.8) | 160.2 | 223.5 (130.4) | 0.39 | 1,959 | yes |
| 17,694,720 | laplace | 12 | 91.0 (20.7) | 94.9 | 218.0 (110.2) | 0.42 | 1,213 | yes |
| 17,694,720 | t3 | 12 | 85.5 (11.3) | 87.6 | 231.0 (95.0) | 0.37 | 1,153 | yes |
| 17,694,720 | normal | 16 | 95.5 (26.8) | 95.7 | 226.7 (74.0) | 0.42 | 30,756 | yes |
| 17,694,720 | laplace | 20 | 86.8 (13.9) | 191.0 | 246.8 (63.6) | 0.35 | 349,161 | yes |
| 17,694,720 | t3 | 20 | 86.4 (15.1) | 110.9 | 261.0 (65.1) | 0.33 | 336,624 | yes |

Bytes: the sort works on one f32 copy (4 B/elem, 70.8 MB at 17.7M). The histogram path also builds a u32 key array (4 B/elem) and casts to f64 inside the weighted bincount, so it uses more memory, not less. All 21 histogram runs (9 at S = 12, 6 each at S = 16 and S = 20; the table shows 15 of them) matched the sort reference bit for bit on all three f32 gammas.

## Verdict

**not-viable.** The bracket is exact (21/21 bit-identical, and only about 0.01% of the tensor needs an in-bin sort at S = 12), but numpy 2.5's vectorized sort already does the whole f32 tensor in about 87 ms for 17.7M weights. Two bincount passes and one mask pass cost 2.4-3x more than that at full size and 1.2-1.8x more at 1-4M, and the bin width does not change it. The 16:45 daily audit should keep sorting once in f32 (Winsor `55ac5fb`'s definitions), take all three scalars from that sort, and store them per the 14:55 note. The stored-scalar win lives in the weekly CUSUM, not in the daily computation.

## This wave's pair files

- Winsor (pair A): JuniorQuant `55ac5fb` `src/daily_audit_1645_f32_gamma_receipt.py`, the f32 gamma receipt whose definitions this bench reproduces.
- Trit Pack (pair A): no 18:20 push seen by 19:05 MT. The last one is BitNet-mlx `ae4644e` (14:20 wave).
- Pair B: void this wave (Chief: no roadmap gap; Handshake JuniorHome `15b570f` i2sd userspace registry landed at 18:10).

## Next question

Can the 16:45 daily ternary quantize reuse the junior_drop/p95 sort it already does? With |W| sorted once, the clip fraction and the in-range RMS error for each gamma can come from `searchsorted` plus prefix sums over the sorted array, with no extra pass over W. Is that at least 1.5x cheaper than Winsor's per-element `ternary_stats` loop vectorized in numpy, with identical clip counts and RMS within 1e-6 relative?

close: d9ff2894f542eb54 (sha256 of previous note text || this note text, first 16 hex; prev 2026-10-23_1455)
