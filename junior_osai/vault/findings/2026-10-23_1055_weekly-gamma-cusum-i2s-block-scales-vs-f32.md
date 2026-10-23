---
local: true
date: 2026-10-23
wave: "2026-10-23 10:05"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: JuniorCloud chief weekly Sun/Mon 17:00 MDT (writes docs/WEEKLY_AUDIT.md), weekly gamma change-point section"
chief_assignment: "Chief 10:05 (quant-chief/last_tips.json 16:10Z): pair A chief weekly audit remaining engines web3node/stocksnode; pair B 16:45 daily audit remaining engines BitNet-mlx/JuniorQuant"
verdict: viable
tags: [ternary, osai, findings, automations, chief-weekly, weekly-audit, winsor, cusum, gamma, i2s, block-scale, numpy, viable]
---

# Weekly gamma CUSUM: the absmean channel runs 68-136x faster from the I2_S 128-block scales than from f32 weights, with identical alarms; junior_drop and p95 still need the f32 weights

## Question

Winsor's change-point check for the chief weekly audit (JuniorQuant `8806d92`, `src/weekly_audit_gamma_cusum.py`) runs a two-sided Page CUSUM (k 0.025, h 0.15, reference = mean of days 0-2) on log(gamma_day / gamma_ref) for absmean, junior_drop and p95, per engine (web3node, stocksnode) and per port. Each day's I2_S pack already stores one f32 absmean scale per 128 weights. Can the weekly CUSUM read those block scales instead of re-reading the f32 weights, be at least 1.5x cheaper, and alarm on the same days?

## Method

Synthetic weights, Student-t(4) x 0.02 in f32 (absmean normalized), seed 1055, 70 tensors (2 engines x 5 ports x 7 days), 262,144 and 1,048,576 weights per tensor, 7 repeats each. Injected scenarios per engine/port: none (4), absmean scale +8% from day 4 (2), scale -5% from day 5 (1), tail-only shift from day 4 (Student-t(2.5) with absmean held fixed, 3). Paths:

- f32 sort (Winsor's shape, vectorized): sort |w|, take mean, mean of the lowest 15/16, and the 95th percentile.
- f32 partition: same gammas via `np.partition` (exact match to sort, max rel diff 8.9e-16).
- block scales: the stored per-128 absmean scales; absmean = mean of scales (exact for equal blocks), junior_drop and p95 proxied by the same statistics over block scales.
- absmean only: one f32 pass `mean(|w|)` against the mean of block scales, the fair baseline for that single channel.

Command (in `/workspace/tbench/r20261023_1055`): `python3 blockscale_cusum_bench.py 262144 7` and `python3 blockscale_cusum_bench.py 1048576 7`.

Inline script:

```python
import sys, time, math, numpy as np
# Weekly gamma CUSUM: f32 weights (reference) vs per-128-block absmean scales already in the I2_S pack.
N=int(sys.argv[1]); R=int(sys.argv[2]); B=128
PORTS=("Home","OSai","FieldCore","Flagstaff","JuniorLLM"); ENG=("web3node","stocksnode")
rng=np.random.default_rng(1055)
# scenario per (engine,port): none / scale +8% from day 4 / tail-only (df4->df2.5, absmean held) from day 4 / scale -5% from day 5
SC=["none","scale+8@4","tail@4","none","scale-5@5","tail@4","none","scale+8@4","none","tail@4"]
def day_w(sc,d):
    if sc.startswith("tail") and d>=4:
        x=rng.standard_t(2.5,N); x*= (np.sqrt(2)/np.pi*0+1)  # placeholder
        x=x/np.mean(np.abs(x))*0.02*0.7978  # match absmean of t4*0.02 approx
    else:
        x=rng.standard_t(4,N)*0.02
        x=x/np.mean(np.abs(x))*0.02*0.7978
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
def g_sort(w):
    a=np.sort(np.abs(w)); n=a.size; keep=n-n//16
    return (a.mean(dtype=np.float64), a[:keep].mean(dtype=np.float64), float(a[int(math.ceil(0.95*n))-1]))
def g_part(w):
    a=np.abs(w); n=a.size; keep=n-n//16; k95=int(math.ceil(0.95*n))-1
    p=np.partition(a,[k95,keep-1])
    return (a.mean(dtype=np.float64), p[:keep].mean(dtype=np.float64), float(p[k95]))
# daily side (pack time): block absmean scales, f32, one per 128 weights
def scales(w): return np.abs(w.reshape(-1,B)).mean(axis=1,dtype=np.float32)
t=[]
for _ in range(R):
    t0=time.perf_counter(); S=[[scales(w) for w in days] for days in W]; t.append(time.perf_counter()-t0)
daily_t=np.array(t)
def g_blk(s):
    n=s.size; keep=n-n//16; k95=int(math.ceil(0.95*n))-1
    p=np.partition(s,[k95,keep-1])
    return (s.mean(dtype=np.float64), p[:keep].mean(dtype=np.float64), float(p[k95]))
def run(fn,data):
    G=np.array([[fn(x) for x in days] for days in data])  # (10,7,3)
    alarms=[[cusum2(list(np.log(G[i,:,j]/G[i,:REF,j].mean()))) for j in range(3)] for i in range(G.shape[0])]
    return G,alarms
res={}
for name,fn,data in (("f32_sort",g_sort,W),("f32_partition",g_part,W),("blockscale",g_blk,S)):
    ts=[]
    for _ in range(R):
        t0=time.perf_counter(); G,al=run(fn,data); ts.append(time.perf_counter()-t0)
    res[name]=(np.array(ts),G,al)
bytes_f32=sum(w.nbytes for d in W for w in d); bytes_blk=sum(s.nbytes for d in S for s in d)
ref=res["f32_partition"][0]
print(f"N={N} tensors=70 repeats={R} f32_bytes={bytes_f32} blockscale_bytes={bytes_blk} ratio={bytes_f32/bytes_blk:.1f}x")
print(f"daily scale build (all 70): median {np.median(daily_t)*1e3:.2f} ms  range {daily_t.min()*1e3:.2f}-{daily_t.max()*1e3:.2f}")
for k,(ts,G,al) in res.items():
    print(f"{k:14s} median {np.median(ts)*1e3:9.2f} ms  range {ts.min()*1e3:.2f}-{ts.max()*1e3:.2f}  speedup_vs_partition {np.median(ref)/np.median(ts):.1f}x  vs_sort {np.median(res['f32_sort'][0])/np.median(ts):.1f}x")
Gf=res["f32_sort"][1]; Gb=res["blockscale"][1]
print(f"absmean rel err blockscale vs f32: max {np.max(np.abs(Gb[:,:,0]/Gf[:,:,0]-1)):.2e}")
print(f"partition vs sort gammas max rel diff: {np.max(np.abs(res['f32_partition'][1]/Gf-1)):.2e}")
print("scenario        | f32 alarm day (absmean,junior_drop,p95) | blockscale alarm day")
agree=0
for i,sc in enumerate(SC):
    a=res["f32_sort"][2][i]; b=res["blockscale"][2][i]; agree+=sum(x==y for x,y in zip(a,b))
    print(f"{ENG[i//5]}/{PORTS[i%5]:9s} {sc:9s} | {a} | {b}")
print(f"alarm agreement {agree}/30 gamma-port cells")
# proxy ratio stability: blockscale p95 / f32 p95 per day
r=Gb[:,:,2]/Gf[:,:,2]; print(f"blockscale p95 / f32 p95 ratio: none-days {r[[0,3,6,8],:].min():.3f}-{r[[0,3,6,8],:].max():.3f}; tail days>=4 {r[[2,5,9],4:].min():.3f}-{r[[2,5,9],4:].max():.3f}")
# absmean-only channel: fair baseline is one f32 pass, no sort
def am_f32():
    G=np.array([[np.abs(w).mean(dtype=np.float64) for w in d] for d in W]); return [cusum2(list(np.log(G[i]/G[i,:REF].mean()))) for i in range(10)]
def am_blk():
    G=np.array([[s.mean(dtype=np.float64) for s in d] for d in S]); return [cusum2(list(np.log(G[i]/G[i,:REF].mean()))) for i in range(10)]
for nm,fn in (("absmean_f32",am_f32),("absmean_blockscale",am_blk)):
    ts=[]
    for _ in range(R):
        t0=time.perf_counter(); al=fn(); ts.append(time.perf_counter()-t0)
    ts=np.array(ts); res[nm]=ts
    print(f"{nm:18s} median {np.median(ts)*1e3:8.3f} ms range {ts.min()*1e3:.3f}-{ts.max()*1e3:.3f} alarms {al}")
print(f"absmean-only speedup blockscale vs f32 one pass: {np.median(res['absmean_f32'])/np.median(res['absmean_blockscale']):.1f}x")
```

## Results

Times are for the whole week (70 tensors), median with min-max over 7 repeats.

| N per tensor | path | median ms | range ms | bytes read | speedup |
|---|---|---|---|---|---|
| 262,144 | f32 sort, 3 gammas | 90.92 | 84.28-110.39 | 73,400,320 | 1.0x |
| 262,144 | f32 partition, 3 gammas | 162.00 | 159.63-171.51 | 73,400,320 | 0.6x |
| 262,144 | block scales, 3 gammas | 1.75 | 1.67-1.92 | 573,440 | 51.9x |
| 262,144 | f32 absmean only | 16.31 | 15.14-17.15 | 73,400,320 | 1.0x |
| 262,144 | block-scale absmean only | 0.239 | 0.232-0.372 | 573,440 | 68.1x |
| 1,048,576 | f32 sort, 3 gammas | 356.44 | 346.18-368.02 | 293,601,280 | 1.0x |
| 1,048,576 | f32 partition, 3 gammas | 646.64 | 625.27-671.64 | 293,601,280 | 0.6x |
| 1,048,576 | block scales, 3 gammas | 6.08 | 5.73-6.78 | 2,293,760 | 58.7x |
| 1,048,576 | f32 absmean only | 69.70 | 67.98-103.83 | 293,601,280 | 1.0x |
| 1,048,576 | block-scale absmean only | 0.512 | 0.464-0.733 | 2,293,760 | 136.1x |

Daily cost of writing the block scales (if the pack did not already): 13.4 ms (262k) and 56.0 ms (1M) for all 70 tensors.

Error and alarm agreement against the f32 sort reference:

| check | 262,144 | 1,048,576 |
|---|---|---|
| absmean rel error, block scales vs f32 | 3.3e-09 | 1.5e-09 |
| absmean alarms identical | 10/10 ports | 10/10 ports |
| junior_drop alarms identical | 7/10 (misses all 3 tail-only shifts) | 7/10 (misses all 3 tail-only shifts) |
| p95 alarms identical | 9/10 (misses web3node/OSai scale +8%) | 10/10 |
| all cells | 26/30 | 27/30 |
| block-scale p95 / f32 p95 ratio | 0.411-0.422 | 0.413-0.417 |

## Verdict

**Viable, for the absmean channel only.** Reading the per-128-block absmean scales that the I2_S pack already stores makes the weekly absmean CUSUM 68x (262k) to 136x (1M) faster than one f32 pass, reads 128x fewer bytes, matches f32 absmean to about 3e-9 relative, and alarms on exactly the same days. It is not a substitute for junior_drop or p95: block means average away the tail, so the proxies missed every tail-only shift that the f32 junior_drop channel caught, and the p95 proxy sat near the alarm threshold on one scale shift. So the weekly audit should take absmean from the pack and keep one f32 sort per tensor for junior_drop and p95 (or have the daily run store those two exact gammas as scalars, which removes the weekly f32 read entirely). Side note for Winsor: `np.partition` with two kth values was 1.8x slower than a full sort here, so keep the sort.

## Links

- Previous note: [[2026-10-23_0555_weekly-clip-rate-1bit-planes-vs-f32-rescan]]
- This wave's pair files: Winsor JuniorQuant `8806d92` `src/weekly_audit_gamma_cusum.py` (chief weekly audit, web3node/stocksnode). No Trit Pack, Side-Commit or Handshake & Ports push was visible in local copies at 10:55.

## Next question

If the daily run stores the exact junior_drop and p95 gammas as two f32 scalars per tensor next to the block scales, does the whole weekly CUSUM (all three channels plus the Bernoulli clip CUSUM from the clip planes) run at least 1.5x cheaper end to end than Winsor's f32 path, with 30/30 identical alarms?

close: 960d84148d12de98 (sha256 of previous note text || this note text, first 16 hex; prev 2026-10-23_0555)
