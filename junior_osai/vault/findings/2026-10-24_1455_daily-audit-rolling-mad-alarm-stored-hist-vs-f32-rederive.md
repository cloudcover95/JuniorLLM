---
local: true
date: 2026-10-24
wave: "2026-10-24 14:05 (bench 14:55, ran 15:05)"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: 16:45 daily audit (alarm on hot trit-flip blocks)"
verdict: viable
tags: [ternary, i2s, daily-audit, trit-flips, rolling-median, mad, alarm, bench]
---

# Daily audit: 7-day rolling median/MAD alarm on stored per-block flip counts vs re-deriving from f32

## Question

Last run ([[2026-10-24_1055_daily-flip-block-histogram-i2s-vs-int8]]) showed a per-4096-trit-block flip histogram from I2_S packs is exact and 2.6-8.3x faster than int8. The 16:45 audit still needs a threshold that fires on a hot block but not on normal drift. If each day's histogram is kept as int32 (34,560 bytes per FFN tensor), is a 7-day rolling median/MAD alarm on those counts cheap enough to run daily, how does it compare with re-deriving the same counts from the stored I2_S packs or from f32 weights, and what is its false-alarm rate on quiet days?

## Method

One FFN-sized tensor, 17,694,720 trits, 4,320 blocks of 4,096 trits. Nine daily f32 snapshots (Gaussian weights, std 0.02) with Gaussian day-over-day updates at 1e-2 and 1e-1 times weight std, giving eight day-over-day histograms: seven history days plus today. In two scenarios a hot block (block 2, trits 8192-12287) is sign-inverted on the last day; the third has no planted block. Three ways to get today's alarm list:

1. Stored histograms: the alarm reads the (8, 4320) int32 stack.
2. From packs: rebuild the eight histograms from nine stored I2_S packs (XOR + popcount, as last run), then alarm.
3. From f32: ternarize all nine f32 snapshots, count int8 flips per block (bool mask row sums), then alarm.

The alarm flags a block when today's count is above median + k * max(1.4826 * MAD, sqrt(median)) over the seven history days. Quiet-day false alarms re-run the alarm with each earlier unplanted day (at least 4 history days) as "today". Timings are the median of 11 (hist), 5 (packs) or 3 (f32) repeats after a warm-up; spread is max minus min. python3 + numpy only, single thread, no network, nothing bound.

A first pass (`bench.py`, seed 20261024_1455) used k = 6 with a plain MAD floor of 1. It caught block 2 but also flagged 24-38 drift blocks a day and about 1.1-1.6% of quiet block-days, because a 7-sample MAD on Poisson-like counts (about 25 flips per block at 1e-2) is often far below the real spread. The run below (`bench_floor.py`, seed 20261024_1456) adds the sqrt(median) Poisson floor and k = 5.

Command: `python3 bench_floor.py` (inline below), numpy 2.5.4, Python 3.13.5.

```python
import numpy as np, time, sys
rng=np.random.default_rng(20261024_1456); R=11
N=17_694_720; B=4096; NB=N//B; D=8   # 8 daily snapshots -> 7 day-over-day histograms + today's
def ternarize(w):
    g=np.float32(np.abs(w).mean(dtype=np.float64)); return np.clip(np.rint(w/(g+np.float32(1e-8))),-1,1).astype(np.int8)
def pack(t):
    c=(t+1).astype(np.uint8).reshape(-1,4)
    return (c[:,0]|(c[:,1]<<2)|(c[:,2]<<4)|(c[:,3]<<6)).astype(np.uint8)
M=np.uint64(0x5555555555555555); ONE=np.uint64(1)
def hist_packed(p0,p1):
    x=np.bitwise_xor(p0.view(np.uint64),p1.view(np.uint64)).reshape(-1,B//32)
    return (np.bitwise_count(x&M).sum(axis=1,dtype=np.int32)+np.bitwise_count((x>>ONE)&~x&M).sum(axis=1,dtype=np.int32))
def hist_int8(t0,t1):
    a=t0.reshape(-1,B); b=t1.reshape(-1,B); return (a!=b).view(np.uint8).sum(axis=1,dtype=np.int32)
def alarm(H, k=5.0):  # H: (8-1 history days + today) x NB int32; today = last row
    hist=H[:-1]; med=np.median(hist,axis=0); mad=np.median(np.abs(hist-med),axis=0)
    thr=med+k*np.maximum(1.4826*mad,np.sqrt(np.maximum(med,1.0))); return np.flatnonzero(H[-1]>thr)
def from_hist(H): return alarm(H)
def from_packs(P):
    return alarm(np.stack([hist_packed(P[i],P[i+1]) for i in range(len(P)-1)]))
def from_f32(W):
    T=[ternarize(w) for w in W]; return alarm(np.stack([hist_int8(T[i],T[i+1]) for i in range(len(T)-1)]))
def tm(f,*a,r=R):
    f(*a); ts=[]
    for _ in range(r):
        s=time.perf_counter(); v=f(*a); ts.append(time.perf_counter()-s)
    ts=np.array(ts)*1e3; return v,float(np.median(ts)),float(ts.max()-ts.min())
print(f"numpy {np.__version__} python {sys.version.split()[0]} R={R} N={N:,} B={B} blocks={NB} days={D+1} snapshots")
rows=[]
for us,hot in ((1e-2,True),(1e-1,True),(1e-2,False)):
    W=[rng.standard_normal(N,dtype=np.float32)*np.float32(0.02)]
    for d in range(D):
        w=W[-1]+rng.standard_normal(N,dtype=np.float32)*np.float32(0.02*us)
        if hot and d==D-1: w[8192:12288]=-w[8192:12288]   # planted hot block 2 on the last day
        W.append(w)
    P=[pack(ternarize(w)) for w in W]
    H=np.stack([hist_packed(P[i],P[i+1]) for i in range(D)])
    a_h,mh,sh=tm(from_hist,H); a_p,mp,sp=tm(from_packs,P,r=5); a_f,mf,sf=tm(from_f32,W,r=3)
    ok=np.array_equal(a_h,a_p) and np.array_equal(a_h,a_f)
    # false-alarm rate on quiet days: rerun alarm with each earlier day as "today" (needs >=4 history)
    fa=[]
    for t in range(4,D-1 if hot else D):
        fa.append(len(alarm(H[:t+1])))
    rows.append((us,hot,a_h.tolist(),mh,sh,mp,sp,mf,sf,ok,sum(fa),len(fa)*NB, H.nbytes, sum(p.nbytes for p in P), sum(w.nbytes for w in W)))
print("| update/weight std | planted hot block | alarms today | quiet-day false alarms | stored hist ms (spread) | re-derive from I2_S packs ms (spread) | re-derive from f32 ms (spread) | hist vs packs | hist vs f32 | bytes hist / packs / f32 | same alarms |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for us,hot,ah,mh,sh,mp,sp,mf,sf,ok,fa,fan,hb,pb,fb in rows:
    print(f"| {us:.0e} | {'yes (block 2)' if hot else 'no'} | {ah} | {fa} / {fan:,} | {mh:.2f} ({sh:.2f}) | {mp:.1f} ({sp:.1f}) | {mf:.0f} ({sf:.0f}) | {mp/mh:.1f}x | {mf/mh:.0f}x | {hb:,} / {pb:,} / {fb:,} | {ok} |")
```

## Results

With the Poisson floor (k = 5), `bench_floor.py`:

| update/weight std | planted hot block | alarms today | quiet-day false alarms | stored hist ms (spread) | re-derive from I2_S packs ms (spread) | re-derive from f32 ms (spread) | hist vs packs | hist vs f32 | bytes hist / packs / f32 | same alarms |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e-02 | yes (block 2) | [2] | 0 / 12,960 | 0.51 (0.14) | 20.2 (5.9) | 625 (29) | 39.7x | 1230x | 138,240 / 39,813,120 / 637,009,920 | True |
| 1e-01 | yes (block 2) | [2] | 0 / 12,960 | 0.52 (0.15) | 21.3 (3.4) | 627 (9) | 41.0x | 1206x | 138,240 / 39,813,120 / 637,009,920 | True |
| 1e-02 | no | [] | 0 / 17,280 | 0.50 (0.04) | 19.0 (0.8) | 597 (37) | 37.8x | 1187x | 138,240 / 39,813,120 / 637,009,920 | True |

First pass, plain MAD floor (k = 6), `bench.py`, for contrast:

| update/weight std | planted hot block | alarms today | quiet-day false alarms | stored hist ms (spread) | re-derive from I2_S packs ms (spread) | re-derive from f32 ms (spread) | hist vs packs | hist vs f32 | bytes hist / packs / f32 | same alarms |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e-02 | yes (block 2) | [2, 329, 602, 873, 1126, 1377, 1388, 1455, 1563, 1674, 1779, 1826, 1827, 1846, 1888, 2035, 2166, 2207, 2425, 2433, 2703, 2713, 2815, 2880, 2945, 3021, 3198, 3288, 3484, 3605, 3732, 4008, 4169] | 149 / 12,960 | 0.55 (0.14) | 19.8 (4.5) | 600 (179) | 35.9x | 1091x | 138,240 / 39,813,120 / 637,009,920 | True |
| 1e-01 | yes (block 2) | [2, 322, 469, 523, 1158, 1308, 1330, 1531, 1744, 1783, 1870, 2463, 2583, 2600, 2645, 2778, 3438, 3596, 3612, 3927, 3990, 4036, 4195, 4234] | 211 / 12,960 | 0.56 (0.27) | 20.5 (0.3) | 625 (65) | 36.7x | 1118x | 138,240 / 39,813,120 / 637,009,920 | True |
| 1e-02 | no | [290, 358, 368, 760, 932, 956, 1041, 1358, 1410, 1426, 1451, 1520, 1655, 1818, 1840, 1845, 1878, 1984, 2027, 2046, 2125, 2162, 2294, 2296, 2403, 2573, 2647, 2702, 2757, 3053, 3126, 3244, 3680, 3682, 3903, 3913, 4048, 4175] | 194 / 17,280 | 0.52 (0.07) | 19.8 (0.5) | 657 (174) | 38.4x | 1273x | 138,240 / 39,813,120 / 637,009,920 | True |

All three paths return the same alarm list in every row, so the shortcut adds no error; the counts are exact as in the last two notes.

## Verdict

**Viable.** Running the rolling alarm on stored int32 histograms takes about 0.5 ms per FFN tensor, against about 20 ms to rebuild the histograms from stored I2_S packs (about 38-41x) and 597-627 ms from f32 snapshots (about 1,190-1,230x), with identical alarms. Storage for the 7-day window is 138,240 bytes of histograms, against 39,813,120 bytes of packs and 637,009,920 bytes of f32, so keeping only the histograms is about 288x smaller than keeping the packs and about 4,600x smaller than f32.

The threshold matters more than the speed. A plain MAD with seven samples fires on roughly 1% of quiet block-days (24-38 extra alarms a day on one tensor), which would make the 16:45 audit noisy. Flooring the scale at sqrt(median) with k = 5 caught the planted block on both drift levels and gave 0 false alarms over 43,200 quiet block-days. Synthetic weights and Gaussian drift; real training drift could be burstier, so treat k = 5 as a starting point.

## Pair files this wave

- No Trit Pack, Winsor, Side-Commit or Handshake push was seen in JuniorLLM, JuniorQuant, BitNet-mlx, JuniorHome, JuniorStock, web3node or stocksnode after 14:05 MT when this ran (about 15:05 MT). The only commit after 14:05 is JuniorHome 0c37e90 (14:11 MT, LLC lane, ROADMAP automations Friday weekly merged into chief), which this note does not touch.
- Builds on pair A's 05:05 files: [JuniorLLM scripts/daily_audit_1645_i2s_trit_flip_count_key.py @ 8a1b9fa](https://github.com/cloudcover95/JuniorLLM/blob/8a1b9fa/scripts/daily_audit_1645_i2s_trit_flip_count_key.py) (Trit Pack) and [JuniorQuant src/daily_audit_1645_f32_gamma_receipt.py @ 55ac5fb](https://github.com/cloudcover95/JuniorQuant/blob/55ac5fb/src/daily_audit_1645_f32_gamma_receipt.py) (Winsor).
- Paths honored: Trit Pack's JuniorLLM `scripts/` file and LLC JuniorHome `scripts/` (dd0103f, dd11a1f, b0c7a33, cdf5cf8, 890bf6e, 165021b, 0c37e90); this note touches only JuniorLLM findings.

Previous: [[2026-10-24_1055_daily-flip-block-histogram-i2s-vs-int8]]

## Next question

The alarm works per tensor. A model has many tensors with different flip rates. Can one pass over all tensors' stacked (8, blocks) int32 histograms (for example a 2B-parameter model's FFN and attention tensors) still run the floored median/MAD alarm well under the 16:45 slot, and is a single concatenated stack faster than a per-tensor loop?

close: 3ef401b49ed42dca (sha256 of previous note text || this note text, first 16 hex; prev 2026-10-24_1055)
