---
local: true
date: 2026-10-26
wave: "2026-10-26 05:05 (bench 05:55, ran 06:00)"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: 16:45 daily audit (alarm on hot trit-flip blocks)"
verdict: not-viable
tags: [ternary, i2s, daily-audit, trit-flips, negative-binomial, leave-today-out, two-sided, rolling-sum, folded-lut, bench]
---

# Daily audit: folding the day-level rescale into the lookup (two-sided leave-today-out trit-flip alarm, BitNet 2B4T shapes)

## Question

Last run ([[2026-10-26_0255_daily-audit-stored-day-means-two-sided-loo-vs-mad]]) found that the two-sided stored-means alarm costs about 4.7 to 5.1 ms, and the remaining cost is the 8-wide float dot per block that rescales each history day to today's level. Can the rescale be folded into the lookup so the two-sided alarm runs near the 3 ms one-sided lookup, with the same hot and quiet catches? Two ways were tried: (a) keep a rolling int32 7-day raw history sum per block at day close and apply one scalar scale to it, and (b) bucket that scale into 48 uint8 steps and precompute int16 cutoff tables indexed directly by the raw sum, so the timed path is two gathers and two compares.

## Method

Same synthetic whole-model generator as the last four notes (210 ternary tensors of BitNet b1.58 2B4T, 508,800 blocks of 4,096 trits, int32 per-block flip histograms from the I2_S XOR+popcount path, 9 days, per-tensor Gamma day factor, optional tensor-wide shared day factor, 6 planted hot and 6 planted quiet blocks on the last day, p = 1e-7). The default seed is new (20261026_555); seeds 20261025_1855 and 777 reproduce last run's draws.

Paths, each a per-tensor loop over the whole model:

1. Floored MAD, one-sided and two-sided (baselines).
2. Leave-today-out NB, de-shared fit, no rescale (the 3 ms one-sided reference).
3. Stored f16 day means rescale, two-sided (last run's best).
4. Stored scalar plus rolling sum, two-sided: today's row mean over the mean of the stored day means gives one scale, applied to the stored int32 rolling sum; no history rows are read.
5. Folded bucket table, two-sided and one-sided: the same scale snapped to the nearest of 48 log-spaced steps between 0.4 and 2.5, then int16 upper and lower cutoff rows indexed by the raw rolling sum.

The rolling sums and day means are written when each day closes and are not timed; the day-close update cost is measured separately (an add-and-revert pair over the whole model). 21 timed repeats after one warm-up, median and max-min spread. The six runs ran three at a time in parallel, so absolute times share a loaded CPU and speedups are within-run. numpy 2.5.4, Python 3.13.5, no network, nothing bound.

Commands, from a scratch dir: `python3 bench.py 20261026_555`, `python3 bench.py 20261025_1855`, `python3 bench.py 777`, and the same three seeds with a second argument `shared`.

```python
import numpy as np, time, math, sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 20261026_555
SHARED=len(sys.argv)>2 and sys.argv[2]=="shared"
rng=np.random.default_rng(seed); R=21; B=4096; D=9; P=1e-7
H_, KV, FF, L = 2560, 640, 6912, 30
shapes=[("q",H_,H_),("k",H_,KV),("v",H_,KV),("o",H_,H_),("gate",H_,FF),("up",H_,FF),("down",FF,H_)]
tensors=[(f"L{l}.{n}",a*b//B) for l in range(L) for n,a,b in shapes]
# 9 days: days 0..7 = yesterday's window (stored phi fitted from its 7 history days 0..6), days 1..8 = today's window
hs=[]; phis=[]; planted=set(); quiet=set(); off=0
for i,(name,nb) in enumerate(tensors):
    lam=float(np.exp(rng.uniform(np.log(2),np.log(400))))
    phi=float(np.exp(rng.uniform(np.log(0.005),np.log(0.1)))); a=1/phi
    G=rng.gamma(a,1/a,size=(D,nb))
    if SHARED:  # tensor-wide day factor, phi_s log-uniform 0.005..0.05
        ps=float(np.exp(rng.uniform(np.log(0.005),np.log(0.05)))); G=G*rng.gamma(1/ps,ps,size=(D,1))
    h=rng.poisson(lam*G).astype(np.int32)
    if i%37==5:
        j=int(rng.integers(nb)); sd=math.sqrt(lam+lam*lam*phi); h[-1,j]+=int(10*sd+20); planted.add(off+j)
    if i%37==20:  # quiet plant: block goes nearly silent today (needs lam >= 50 to be detectable)
        lamq=max(lam,50.0); h=rng.poisson(lamq*G).astype(np.int32); j=int(rng.integers(nb)); h[-1,j]=int(rng.integers(0,2)); quiet.add(off+j)
    hs.append(h); phis.append(phi); off+=nb
NBLK=off; CAP=1<<12
lg=np.array([math.lgamma(i+1) for i in range(CAP+1)]); K=np.arange(CAP+1)
def cut_rows(logpmf):
    m=logpmf.max(axis=1,keepdims=True); pm=np.exp(logpmf-m); pm/=pm.sum(axis=1,keepdims=True)
    tail=np.cumsum(pm[:,::-1],axis=1)[:,::-1]; ok=tail<=P
    first=np.where(ok.any(axis=1),ok.argmax(axis=1),CAP+1); return (first-1).astype(np.int32)
def build(phi):
    if phi>0:
        a=1/phi; b=7*a
        LA=np.array([math.lgamma(k+a) for k in K]); LB=np.array([math.lgamma(k+b) for k in K])
        LAB=np.array([math.lgamma(n+a+b) for n in K]); c=math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)
    out=np.empty(CAP+1,np.int32)
    for s in range(0,CAP+1,512):
        N=K[s:s+512,None]; k=K[None,:]; valid=k<=N; nk=np.clip(N-k,0,CAP)
        if phi>0: lp=lg[N]-lg[k]-lg[nk]+LA[None,:]+LB[nk]-LAB[N]+c
        else: lp=lg[N]-lg[k]-lg[nk]+k*math.log(1/8)+nk*math.log(7/8)
        out[s:s+512]=cut_rows(np.where(valid,lp,-np.inf))
    return out
GRID=np.concatenate([[0.0],np.geomspace(0.002,0.32,16)])
t0=time.perf_counter(); LUT=np.stack([build(float(g)) for g in GRID]); t_build=time.perf_counter()-t0
# leave-today-out predictive: today X given 7-day history total S, moment-matched negative binomial
# mean mu=S/7, var V=(mu+phi*mu^2)*(1+1/7) (day-factor + Poisson noise, plus uncertainty of the 7-day mean)
def build_loo(phi):
    out=np.empty(CAP+1,np.int32); k=K[None,:]
    for s in range(0,CAP+1,512):
        S=K[s:s+512,None].astype(np.float64); mu=np.maximum(S,0.5)/7
        V=(mu+phi*mu*mu)*(8/7); r=mu*mu/np.maximum(V-mu,1e-12); q=mu/V  # NB(r, success prob q)
        lp=(np.vectorize(math.lgamma)(k+r)-lg[k]-np.vectorize(math.lgamma)(r)+r*np.log(q)+k*np.log1p(-q))
        out[s:s+512]=cut_rows(lp)
    return out
def cut_lo(logpmf):
    m=logpmf.max(axis=1,keepdims=True); pm=np.exp(logpmf-m); pm/=pm.sum(axis=1,keepdims=True)
    cdf=np.cumsum(pm,axis=1); ok=cdf>P
    return ok.argmax(axis=1).astype(np.int32)  # alarm if x < lo (P(X<lo) <= P)
def build_loo_lo(phi):
    out=np.empty(CAP+1,np.int32); k=K[None,:]
    for s in range(0,CAP+1,512):
        S=K[s:s+512,None].astype(np.float64); mu=np.maximum(S,0.5)/7
        V=(mu+phi*mu*mu)*(8/7); r=mu*mu/np.maximum(V-mu,1e-12); q=mu/V
        lp=(np.vectorize(math.lgamma)(k+r)-lg[k]-np.vectorize(math.lgamma)(r)+r*np.log(q)+k*np.log1p(-q))
        out[s:s+512]=cut_lo(lp)
    return out
t0=time.perf_counter(); LOO=np.stack([build_loo(float(g)) for g in GRID]); t_build2=time.perf_counter()-t0
LOOLO=np.stack([build_loo_lo(float(g)) for g in GRID])
def fit_bin(hist):
    m=hist.mean(axis=0); v=hist.var(axis=0,ddof=1); mm=float(m.mean())
    return max((float(v.mean())-mm)/max(float((m*m).mean()),1e-9),0.0)
def gidx(phi): return min(int(np.searchsorted(GRID,phi*1.1)) if phi>0 else 0,len(GRID)-1)
def gnear(phi): return int(np.abs(GRID-phi).argmin())
W=[h[1:] for h in hs]
def fit_desh(hist):
    f=hist.mean(axis=1,keepdims=True); f=f/max(float(f.mean()),1e-9)  # per-day tensor-wide factor estimate
    return fit_bin(hist/np.maximum(f,1e-9))
phiD=np.array([fit_desh(h[0:7].astype(np.float64)) for h in hs],np.float16)
d8=np.array([gnear(float(p)) for p in phiD],np.uint8)
phi16=np.array([fit_bin(h[0:7]) for h in hs],np.float16)
g8=np.array([gidx(float(p)) for p in phi16],np.uint8); n8=np.array([gnear(float(p)) for p in phi16],np.uint8)
MS=[h[1:8].mean(axis=1).astype(np.float16) for h in hs]  # 7 float16 per tensor, written when each day closes
MSf=[m.astype(np.float64) for m in MS]
def mad_sort(h):
    hist,today=h[:-1],h[-1]; s=np.sort(hist,axis=0); med=s[3].astype(np.float64)
    d=np.sort(np.abs(hist-med),axis=0); mad=d[3]
    return today>med+5.0*np.maximum(1.4826*mad,np.sqrt(np.maximum(med,1.0)))
def inwin(h,i): return h[-1]>LUT[g8[i]][np.minimum(h.sum(axis=0),CAP)]
def loo_up(h,i): return h[-1]>LOO[g8[i]][np.minimum(h[:-1].sum(axis=0),CAP)]
def loo_near(h,i): return h[-1]>LOO[n8[i]][np.minimum(h[:-1].sum(axis=0),CAP)]
def loo_desh(h,i): return h[-1]>LOO[d8[i]][np.minimum(h[:-1].sum(axis=0),CAP)]
def loo_desh_rs(h,i):
    m=h.mean(axis=1); c=m[-1]/np.maximum(m[:-1],1e-9)  # rescale each history day to today's tensor-wide level
    S=np.rint(c@h[:-1]).astype(np.int64)
    return h[-1]>LOO[d8[i]][np.minimum(S,CAP)]
def loo_stored(h,i):
    c=float(h[-1].mean())/np.maximum(MSf[i],1e-9)  # only today's row mean is computed
    S=np.rint(c@h[:-1]).astype(np.int64)
    return h[-1]>LOO[d8[i]][np.minimum(S,CAP)]
def loo_scalar(h,i):
    c=float(h[-1].mean())/max(float(MSf[i].mean()),1e-9)  # one scale for the summed history
    S=np.rint(c*h[:-1].sum(axis=0)).astype(np.int64)
    return h[-1]>LOO[d8[i]][np.minimum(S,CAP)]
def loo_stored_2s(h,i):
    c=float(h[-1].mean())/np.maximum(MSf[i],1e-9)
    S=np.minimum(np.rint(c@h[:-1]).astype(np.int64),CAP); x=h[-1]
    return (x>LOO[d8[i]][S])|(x<LOOLO[d8[i]][S])
def mad_2s(h):
    hist,today=h[:-1],h[-1]; s=np.sort(hist,axis=0); med=s[3].astype(np.float64)
    d=np.sort(np.abs(hist-med),axis=0); mad=d[3]; w=5.0*np.maximum(1.4826*mad,np.sqrt(np.maximum(med,1.0)))
    return (today>med+w)|(today<med-w)
def run(f):
    out=[];o=0
    for i,h in enumerate(W): out.append(np.flatnonzero(f(h,i))+o); o+=h.shape[1]
    return np.concatenate(out)
def tm(f):
    v=run(f); ts=[]
    for _ in range(R):
        s=time.perf_counter(); run(f); ts.append(time.perf_counter()-s)
    ts=np.array(ts)*1e3; return v,float(np.median(ts)),float(ts.max()-ts.min())

# NEW 0555: fold the rescale into the lookup.
# Stored at day close (untimed): rolling int32 7-day raw history sum per block (SR), and the f16 day means (MS).
SR=[h[:-1].sum(axis=0).astype(np.int32) for h in W]
MSbar=np.array([float(m.astype(np.float64).mean()) for m in MS])
def loo_scalar_2s_sr(h,i):  # scalar rescale with stored rolling sum: no history read at all
    c=float(h[-1].mean())/max(MSbar[i],1e-9)
    S=np.minimum(np.rint(c*SR[i]).astype(np.int64),CAP); x=h[-1]
    return (x>LOO[d8[i]][S])|(x<LOOLO[d8[i]][S])
NBK=int(sys.argv[3]) if len(sys.argv)>3 else 48
CB=np.geomspace(0.4,2.5,NBK)
t0=time.perf_counter()
idx=np.minimum(np.rint(CB[:,None]*K[None,:]).astype(np.int64),CAP)  # NBK x 4097
used=sorted(set(d8.tolist()))
FH={g:LOO[g][idx].astype(np.int16) for g in used}; FL={g:LOOLO[g][idx].astype(np.int16) for g in used}
t_fold=time.perf_counter()-t0; fold_bytes=sum(v.nbytes for v in FH.values())+sum(v.nbytes for v in FL.values())
def bucket(c): return min(int(np.abs(np.log(CB)-math.log(max(c,1e-9))).argmin()),NBK-1)
def folded_2s(h,i):  # today's mean -> uint8 scale bucket -> two int16 table rows indexed by the raw stored sum
    b=bucket(float(h[-1].mean())/max(MSbar[i],1e-9)); g=int(d8[i]); s=np.minimum(SR[i],CAP); x=h[-1]
    return (x>FH[g][b][s])|(x<FL[g][b][s])
def folded_1s(h,i):
    b=bucket(float(h[-1].mean())/max(MSbar[i],1e-9)); s=np.minimum(SR[i],CAP)
    return h[-1]>FH[int(d8[i])][b][s]
# day-close cost of keeping SR (subtract the day leaving, add the day entering), whole model
def close_update():
    for i,h in enumerate(W): SR[i]+=h[-2]-h[0]; SR[i]-=h[-2]-h[0]
ts=[]
for _ in range(R):
    s=time.perf_counter(); close_update(); ts.append(time.perf_counter()-s)
t_close=float(np.median(ts))*1e3
res={n:tm(f) for n,f in [("floored MAD, one-sided",lambda h,i:mad_sort(h)),("LOO near, de-shared, no rescale",loo_desh),("floored MAD, two-sided",lambda h,i:mad_2s(h)),("stored f16 means rescale, 2s",loo_stored_2s),("stored scalar + rolling sum, 2s",loo_scalar_2s_sr),("folded bucket table, 2s",folded_2s),("folded bucket table, 1s",folded_1s)]}
base=res["floored MAD, two-sided"][1]
print(f"numpy {np.__version__} seed {seed} shared {SHARED} blocks {NBLK} buckets {NBK} LOO build {t_build2*1e3:.0f} ms fold build {t_fold*1e3:.0f} ms fold tables {fold_bytes} B ({len(used)} grids) rolling sums {sum(s.nbytes for s in SR)} B close update {t_close:.2f} ms")
for n,(v,med,spr) in res.items():
    vs=set(v.tolist()); fa=len(vs-planted-quiet); caught=len(planted&vs); cq=len(quiet&vs)
    print(f"{n:34s} {med:7.2f} ms (spread {spr:5.2f})  {base/med:5.2f}x vs 2s MAD  alarms {len(v):5d}  false {fa:5d}  hot {caught}/{len(planted)}  quiet {cq}/{len(quiet)}")
a=set(res['stored f16 means rescale, 2s'][0].tolist()); b=set(res['folded bucket table, 2s'][0].tolist())
print("folded vs stored-means alarm sets equal:",a==b," only-folded",len(b-a)," only-stored",len(a-b))
```

## Results

Timings in ms, median (spread over 21 repeats). Speedup is against two-sided floored MAD in the same run.

| run | MAD two-sided | stored f16 means, 2s (last run) | stored scalar + rolling sum, 2s | folded bucket table, 2s | folded, 1s | no-rescale 1s reference |
|---|---|---|---|---|---|---|
| 0555 | 38.81 (6.75), 2,245 false, 4/6 quiet | 4.89 (2.05), 7.93x, 0 false, 6/6 hot, 5/6 quiet | 3.55 (6.33), 10.92x, 0 false, 6/6, 5/6 | 5.02 (1.06), 7.73x, same set | 3.46 (4.88) | 3.07, 0 false |
| 1855 | 40.16 (8.67), 2,434 false, 5/6 | 4.66 (1.60), 8.62x, 0 false, 6/6, 6/6 | 3.39 (0.35), 11.86x, 0 false, 6/6, 6/6 | 4.79 (0.26), 8.38x, same set | 3.26 (0.36) | 3.06, 0 false |
| 777 | 39.80 (6.15), 2,529 false, 3/6 | 5.00 (3.42), 7.95x, 0 false, 6/6, 5/6 | 3.47 (0.54), 11.47x, 0 false, 6/6, 5/6 | 5.28 (2.93), 7.54x, same set | 3.63 (0.67) | 3.10, 0 false |
| 0555 shared | 38.05 (6.21), 4,531 false, 5/6 | 4.76 (1.45), 8.00x, 0 false, 6/6, 6/6 | 3.39 (0.34), 11.24x, 0 false, 6/6, 6/6 | 4.83 (0.35), 7.88x, same set | 3.21 (0.26) | 3.03, 151 false |
| 1855 shared | 37.56 (5.66), 4,410 false, 4/6 | 4.75 (1.38), 7.91x, 0 false, 6/6, 6/6 | 3.38 (1.15), 11.12x, 0 false, 6/6, 6/6 | 4.83 (1.06), 7.78x, same set | 3.31 (0.25) | 3.07, 84 false |
| 777 shared | 37.69 (4.55), 6,063 false, 2/6 | 4.88 (2.86), 7.72x, 0 false, 6/6, 6/6 | 3.39 (0.42), 11.11x, 0 false, 6/6, 6/6 | 4.83 (0.53), 7.80x, same set | 3.28 (1.68) | 3.07, 57 false |
| **total** | **24,212 false, 23/36 quiet** | **0 false, 36/36 hot, 34/36 quiet** | **0 false, 36/36 hot, 34/36 quiet** | **identical alarm sets to stored means in all 6** | | **292 false in shared runs** |

Extra state: rolling sums 2,035,200 bytes (int32, one per block); the add-and-revert day-close pair costs 0.65 to 0.72 ms for the whole model, so about half that per real day close. Folded tables: 7.1 to 9.4 MB (int16, only the 9 to 12 grid points actually used), built in under 10 ms from the existing cutoff tables.

The folded bucket tables give exactly the same alarms as last run's stored-means path in all six runs, but are no faster (4.79 to 5.28 ms against 4.66 to 5.00 ms): the gathers now land in a 9 MB table instead of a 278 KB one, and cache misses eat the saved multiply. The simpler stored scalar plus rolling sum path is the real improvement: 3.38 to 3.55 ms two-sided, within about 0.3 to 0.5 ms of the one-sided no-rescale lookup, with the same hot and quiet catch counts and 0 false alarms in about 3.05 million block-days. That is 1.36 to 1.47x faster than last run's stored-means path, which is under the 1.5x bar for this increment.

## Verdict

**Not viable** as an increment. Folding the rescale into bucketed tables is not worth it (no speedup, about 9 MB of extra tables). Storing a rolling int32 sum and using one scalar scale is a useful 1.36 to 1.47x over last run's path for 2 MB of state and an under-1 ms day-close update, but it stays below the 1.5x bar. The two-sided leave-today-out alarm as a whole remains 10.9 to 11.9x faster than two-sided floored MAD with 0 false alarms against 24,212 and 34 of 36 quiet plants against 23 of 36, which restates the earlier viable finding rather than adding a new one. Synthetic data only.

## Pair files this wave

- No pair report reached this run. JuniorLLM's tip at clone time was the previous findings note [359d489](https://github.com/cloudcover95/JuniorLLM/commit/359d489), so there is no new JuniorLLM pair file this wave. This still builds on pair A's 2026-10-24 05:05 files: [JuniorLLM scripts/daily_audit_1645_i2s_trit_flip_count_key.py @ 8a1b9fa](https://github.com/cloudcover95/JuniorLLM/blob/8a1b9fa/scripts/daily_audit_1645_i2s_trit_flip_count_key.py) (Trit Pack) and [JuniorQuant src/daily_audit_1645_f32_gamma_receipt.py @ 55ac5fb](https://github.com/cloudcover95/JuniorQuant/blob/55ac5fb/src/daily_audit_1645_f32_gamma_receipt.py) (Winsor).
- Paths honored: rails/additive/ (c5a1b99) and the JuniorHome paths flagged in recent ledgers are untouched; this note touches only JuniorLLM junior_osai/vault/findings/.

Previous: [[2026-10-26_0255_daily-audit-stored-day-means-two-sided-loo-vs-mad]]

## Next question

The 16:45 daily-audit alarm line is now well covered (five notes in a row). Rotate off it: next run should pick an uncovered ternary question tied to a different ROADMAP.md or BETA_TO_OS.md line, for example whether the rolling-sum state can live in the same I2_S-packed audit receipt the Winsor JuniorQuant f32 gamma receipt already writes, or a fresh decode or matmul question on a line not yet benched.

close: 7cb16e66dc40ae8c (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-26_0255)
