---
local: true
date: 2026-10-23
wave: "2026-10-23 14:05"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: 16:45 daily audit (remaining engines BitNet-mlx/JuniorQuant), gamma scalars stored for the JuniorCloud chief weekly CUSUM"
chief_assignment: "Chief 14:05 (quant-chief/last_tips.json 20:10Z): pair A 16:45 daily audit remaining engines BitNet-mlx/JuniorQuant; pair B chief weekly audit remaining engines web3node/stocksnode"
verdict: viable
tags: [ternary, osai, findings, automations, daily-audit-1645, chief-weekly, winsor, cusum, gamma, i2s, fp32, fp64, numpy, viable]
---

# 16:45 daily audit stores three exact f32 gamma scalars next to the I2_S block scales: the weekly CUSUM gets 1,000-2,800x faster and reads 87,000-350,000x fewer bytes, with 30/30 identical alarms

## Question

The 10:55 note found that I2_S 128-block scales can only stand in for absmean. junior_drop and p95 still need the f32 weights. The 16:45 daily audit already computes all three gammas per engine and port. Winsor's 14:20 file (JuniorStock `c9adc04`) computes them in fp32 (BitNet-mlx) and fp64 (JuniorQuant). If that daily run stores the exact absmean, junior_drop and p95 as three f32 scalars (12 bytes per tensor) beside the block scales, does the chief weekly CUSUM over all three channels run at least 1.5x cheaper than re-sorting a week of f32 weights? Does it alarm on the same days (30/30 cells)? Do f32 scalars match the f64 reference closely enough that the BitNet-mlx/JuniorQuant precision split never changes an alarm?

## Method

The weights are synthetic Student-t(4), normalized to absmean 0.016, stored as f32, seed 1455. There are 70 tensors (2 engines BitNet-mlx/JuniorQuant x 5 ports x 7 days) at 262,144 and 1,048,576 weights each. Each timing has 7 repeats; I report the median and the min-max range. The ten engine/port slots have these injected scenarios: none (4), absmean scale +8% from day 4 (2), scale -5% from day 5 (1), and a tail-only shift from day 4 (3). The tail-only shift is Student-t(2.5) with absmean held fixed. The CUSUM is the same two-sided Page CUSUM as the 10:55 note: k 0.025, h 0.15, reference = mean of days 0-2, run on log(gamma_day / gamma_ref).

- **A (weekly f64 re-sort, Winsor/JuniorQuant shape):** read all 70 f32 tensors, cast to f64, sort |w|, take absmean, the mean of the lowest 15/16 and nearest-rank p95, then run the CUSUM.
- **A32:** the same path in f32 arithmetic (BitNet-mlx precision).
- **B (weekly from stored scalars):** deserialize 3 f32 scalars per tensor (840 bytes for the week) and run the CUSUM. B64 does the same with f64 scalars.
- **Daily cost:** for one day (10 tensors), the time to build the block scales alone, and the time for block scales plus the three f32 gammas from one sort.

Command (in `/workspace/tbench/r20261023_1455`): `python3 daily_gamma_scalars_bench.py 262144 7` and `python3 daily_gamma_scalars_bench.py 1048576 7`.

Inline script:

```python
import sys, time, math, numpy as np
# 16:45 daily audit stores exact gamma scalars (absmean, junior_drop, p95) per tensor next to the
# I2_S 128-block scales; chief weekly CUSUM reads 3 scalars/tensor instead of re-reading f32 weights.
# Also: scalars computed in f32 (BitNet-mlx) vs f64 (JuniorQuant) -> same alarms?
N=int(sys.argv[1]); R=int(sys.argv[2]); B=128
PORTS=("Home","OSai","FieldCore","Flagstaff","JuniorLLM"); ENG=("BitNet-mlx","JuniorQuant")
rng=np.random.default_rng(1455)
SC=["none","scale+8@4","tail@4","none","scale-5@5","tail@4","none","scale+8@4","none","tail@4"]
def day_w(sc,d):
    df=2.5 if (sc.startswith("tail") and d>=4) else 4.0
    x=rng.standard_t(df,N); x=x/np.mean(np.abs(x))*0.016
    if sc.startswith("scale"):
        f=float(sc[5:].split("@")[0])/100; at=int(sc.split("@")[1])
        if d>=at: x=x*(1+f)
    return x.astype(np.float32)
W=[[day_w(sc,d) for d in range(7)] for sc in SC]
K,H,REF=0.025,0.15,3
def cusum2(xs):
    sp=sn=0.0
    for i,x in enumerate(xs):
        sp=max(0.0,sp+x-K); sn=max(0.0,sn-x-K)
        if sp>H or sn>H: return i
    return None
def g_sort(w,dt):
    a=np.sort(np.abs(w.astype(dt,copy=False))); n=a.size; keep=n-n//16
    return (float(a.mean(dtype=dt)), float(a[:keep].mean(dtype=dt)), float(a[int(math.ceil(0.95*n))-1]))
def alarms(G):
    G=np.asarray(G,dtype=np.float64)  # (10,7,3)
    return [[cusum2(list(np.log(G[i,:,j]/G[i,:REF,j].mean()))) for j in range(3)] for i in range(G.shape[0])]
def tm(fn):
    ts=[]
    for _ in range(R):
        t0=time.perf_counter(); out=fn(); ts.append(time.perf_counter()-t0)
    return np.array(ts),out
# A: Winsor's weekly path today: read all 70 f32 tensors, sort, 3 gammas, CUSUM
tA,(GA,alA)=tm(lambda: (lambda G:(G,alarms(G)))([[g_sort(w,np.float64) for w in d] for d in W]))
# A32: same in f32 arithmetic (BitNet-mlx precision)
tA32,(GA32,alA32)=tm(lambda: (lambda G:(G,alarms(G)))([[g_sort(w,np.float32) for w in d] for d in W]))
# daily side: block scales (already in pack) + 3 exact f32 gamma scalars, one day = 10 tensors
def daily(day,dt):
    rec=[]
    for i in range(10):
        w=W[i][day]; s=np.abs(w.reshape(-1,B)).mean(axis=1,dtype=np.float32)
        g=np.array(g_sort(w,dt),dtype=np.float32)
        rec.append((s,g))
    return rec
def daily_scales_only(day):
    return [np.abs(W[i][day].reshape(-1,B)).mean(axis=1,dtype=np.float32) for i in range(10)]
tD,_=tm(lambda: daily(6,np.float32)); tD0,_=tm(lambda: daily_scales_only(6))
REC32=[[None]*7 for _ in range(10)]; REC64=[[None]*7 for _ in range(10)]
for d in range(7):
    for i,(s,g) in enumerate(daily(d,np.float32)): REC32[i][d]=g.tobytes()
    for i,(s,g) in enumerate(daily(d,np.float64)): REC64[i][d]=np.array(g,dtype=np.float64).tobytes()
# B: weekly from stored scalars (deserialize 12 bytes/tensor, CUSUM)
tB,(GB,alB)=tm(lambda: (lambda G:(G,alarms(G)))([[np.frombuffer(r,dtype=np.float32) for r in row] for row in REC32]))
tB64,(GB64,alB64)=tm(lambda: (lambda G:(G,alarms(G)))([[np.frombuffer(r,dtype=np.float64) for r in row] for row in REC64]))
bytes_f32=sum(w.nbytes for d in W for w in d); bytes_s32=sum(len(r) for row in REC32 for r in row); bytes_s64=sum(len(r) for row in REC64 for r in row)
def cmp(a,b): return sum(x==y for p,q in zip(a,b) for x,y in zip(p,q))
GA=np.array(GA); GB=np.array(GB,dtype=np.float64); GA32=np.array(GA32)
print(f"N={N} tensors=70 repeats={R} f32_weight_bytes={bytes_f32} scalar_bytes_f32={bytes_s32} scalar_bytes_f64={bytes_s64} ratio={bytes_f32/bytes_s32:.0f}x")
def row(nm,ts,base=None):
    s=f"{nm:34s} median {np.median(ts)*1e3:10.3f} ms range {ts.min()*1e3:.3f}-{ts.max()*1e3:.3f}"
    if base is not None: s+=f" speedup {np.median(base)/np.median(ts):.1f}x"
    print(s)
row("A weekly f64 sort (Winsor path)",tA,tA); row("A32 weekly f32 sort",tA32,tA)
row("B weekly from f32 scalars",tB,tA); row("B64 weekly from f64 scalars",tB64,tA)
row("daily scales only (10 tensors)",tD0); row("daily scales + f32 gammas (10)",tD)
print(f"daily added cost per day {(np.median(tD)-np.median(tD0))*1e3:.2f} ms; x7 = {(np.median(tD)-np.median(tD0))*7e3:.2f} ms")
print(f"end-to-end A vs (7*daily_added + B): {np.median(tA)/((np.median(tD)-np.median(tD0))*7+np.median(tB)):.2f}x")
print(f"end-to-end A32 (f32 weekly) vs (7*daily_added + B): {np.median(tA32)/((np.median(tD)-np.median(tD0))*7+np.median(tB)):.2f}x")
print(f"f32 scalar rel err vs f64: absmean {np.max(np.abs(GB[:,:,0]/GA[:,:,0]-1)):.2e} junior_drop {np.max(np.abs(GB[:,:,1]/GA[:,:,1]-1)):.2e} p95 {np.max(np.abs(GB[:,:,2]/GA[:,:,2]-1)):.2e}")
print(f"alarm agreement: B(f32 scalars) vs A {cmp(alB,alA)}/30; B64 vs A {cmp(alB64,alA)}/30; A32 vs A {cmp(alA32,alA)}/30")
for i,sc in enumerate(SC): print(f"{ENG[i//5]}/{PORTS[i%5]:9s} {sc:9s} A {alA[i]} B {alB[i]}")
```

## Results

All times are for the whole week (70 tensors) unless marked daily.

| N per tensor | path | median ms | range ms | bytes read weekly | speedup vs A |
|---|---|---|---|---|---|
| 262,144 | A weekly f64 re-sort | 181.09 | 180.04-182.73 | 73,400,320 | 1.0x |
| 262,144 | A32 weekly f32 re-sort | 71.42 | 70.75-72.64 | 73,400,320 | 2.5x |
| 262,144 | B weekly from f32 scalars | 0.171 | 0.169-0.260 | 840 | 1,057x |
| 262,144 | B64 weekly from f64 scalars | 0.176 | 0.167-0.213 | 1,680 | 1,032x |
| 1,048,576 | A weekly f64 re-sort | 628.68 | 615.13-636.64 | 293,601,280 | 1.0x |
| 1,048,576 | A32 weekly f32 re-sort | 311.66 | 305.84-318.43 | 293,601,280 | 2.0x |
| 1,048,576 | B weekly from f32 scalars | 0.223 | 0.168-0.307 | 840 | 2,819x |
| 1,048,576 | B64 weekly from f64 scalars | 0.185 | 0.167-0.197 | 1,680 | 3,400x |

Daily side (one day, 10 tensors):

| N per tensor | block scales only | block scales + 3 f32 gammas | added per day | added x 7 days |
|---|---|---|---|---|
| 262,144 | 1.24 ms (1.20-1.26) | 11.01 ms (10.66-11.70) | 9.77 ms | 68.4 ms |
| 1,048,576 | 5.33 ms (5.21-5.55) | 47.57 ms (46.97-51.52) | 42.24 ms | 295.7 ms |

End to end, counting the full daily gamma cost (A / (7 x added + B)): 2.64x / 2.12x against the f64 weekly path, but only **1.04x / 1.05x against the f32 weekly path**.

Error and alarms (all against A, the f64 reference):

| check | 262,144 | 1,048,576 |
|---|---|---|
| f32 scalar rel err, absmean | 7.8e-08 | 1.6e-07 |
| f32 scalar rel err, junior_drop | 1.0e-07 | 8.7e-08 |
| f32 scalar rel err, p95 | 0 (exact order statistic) | 0 |
| alarms, B (f32 scalars) vs A | 30/30 | 30/30 |
| alarms, B64 vs A | 30/30 | 30/30 |
| alarms, A32 vs A | 30/30 | 30/30 |

Both sizes give the same alarm pattern. Both scale +8% slots alarm on day 6 in all three channels. All three tail-only slots alarm on day 6 in junior_drop only. The scale -5% shift from day 5 does not cross h by day 6 (true for A and B alike). The null slots never alarm.

## Verdict

**Viable** for the weekly read and memory, with one caveat on total compute.

- Reading three stored f32 gamma scalars per tensor makes the chief weekly CUSUM 1,000-2,800x faster and cuts its weekly read from 73-294 MB of f32 weights to 840 bytes. It also means the weekly no longer has to keep 7 days of weights at all.
- Unlike the 10:55 block-scale proxy, all three channels are covered. Alarms match on 30/30 cells, including every tail-only shift.
- The worst f32 vs f64 gap is 1.6e-07 relative (p95 is exact), and it never moves an alarm. So the BitNet-mlx (fp32) vs JuniorQuant (fp64) split is harmless to the weekly CUSUM. This is consistent with the bound in Winsor's `c9adc04`.

**Caveat:** the saving is real only because the 16:45 daily audit already computes these gammas. If the daily run had to sort the weights just to produce them, total compute would be only 1.04-1.05x better than an f32 weekly re-sort. That is not viable on compute alone, though the memory gain still holds. Corollary for JuniorQuant: an f32 sort is 2.0-2.5x faster than its f64 cast-and-sort, with identical alarms.

## Links

- Previous note: [[2026-10-23_1055_weekly-gamma-cusum-i2s-block-scales-vs-f32]]
- This wave's pair files (pair A, 16:45 daily audit BitNet-mlx/JuniorQuant, seen in local clones and on origin/main):
  - Trit Pack & I2_S: BitNet-mlx `ae4644e` `scripts/daily_audit_1645_bitnet_mlx_roundtrip_i2s_key.py` (14:26 MT)
  - Winsor & Quant Error: JuniorStock `c9adc04` `quant/daily_audit_1645_fp32_gamma_agreement.py` (14:31 MT)
- Pair B (Side-Commit / Handshake & Ports, chief weekly web3node/stocksnode): no 14:35 push was visible locally at 14:55.

## Next question

Can the 16:45 daily audit get the exact junior_drop and p95 without a full sort? The candidate is one f32 pass that histograms |w| into fine bins (or into bins on the f32 exponent and top mantissa bits) to bracket the 15/16 and 95% ranks, followed by an exact sort or partition of only the bracketed values (about 1% of the tensor). The 10:55 note found a two-kth `np.partition` slower than a full sort, so that is not the route. The bar is at least 1.5x less daily compute than the f32 sort, with bit-identical gammas. If it clears, the stored-scalar path also wins end to end when the daily run does not already sort.

close: 57128fcecb56bc8c (sha256 of previous note text || this note text, first 16 hex; prev 2026-10-23_1055)
