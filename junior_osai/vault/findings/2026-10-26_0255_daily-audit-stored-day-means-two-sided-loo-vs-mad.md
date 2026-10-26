---
local: true
date: 2026-10-26
wave: "2026-10-26 02:05 (bench 02:55, ran 02:57)"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: 16:45 daily audit (alarm on hot trit-flip blocks)"
verdict: viable
tags: [ternary, i2s, daily-audit, trit-flips, negative-binomial, leave-today-out, shared-day-factor, stored-means, two-sided, float16, bench]
---

# Daily audit: stored float16 day means and a two-sided leave-today-out trit-flip alarm (BitNet 2B4T shapes)

## Question

Last run ([[2026-10-25_1855_daily-audit-deshared-phi-rescaled-loo-vs-floored-mad]]) got 35 of 36 planted hot blocks at 1 false alarm with the de-shared phi fit plus history rescale, but the rescale cost about 2.5 ms on top of the 3 ms lookup. If the per-day tensor-wide means are stored as float16 when each day closes (so only today's row mean is computed), does the rescaled path drop back near 3 ms with the same alarms? And does a two-sided version, which also alarms on blocks that suddenly go quiet, cost anything extra compared with a two-sided floored MAD?

## Method

Same synthetic whole-model generator as the last three notes (210 ternary tensors of BitNet b1.58 2B4T, 508,800 blocks of 4,096 trits, int32 per-block flip histograms from the I2_S XOR+popcount path, 9 days, per-tensor Gamma day factor, optional tensor-wide shared day factor, 6 planted hot blocks on the last day). New this run: 6 planted quiet blocks (one per tensor where i % 37 == 20, that tensor's rate floored at 50 flips per day, today's count set to 0 or 1). The extra draws shift the random stream, so seeds 20261025_1855 and 777 no longer reproduce last run's exact draws; 20261026_255 is new.

Ten per-tensor-loop paths at p = 1e-7. Paths 1 to 6 are last run's. New:

7. Rescale with stored means: 7 float16 tensor means per tensor (2,940 bytes for the model window), written untimed; only `h[-1].mean()` and the 8-wide dot are timed.
8. Rescale with one stored scalar: history total times today's mean over the mean of the stored means.
9. Two-sided floored MAD (also alarms below median minus the same width).
10. Two-sided stored rescale: path 7 plus a second 17 x 4,097 int32 lower-cutoff table (alarm when today is below the smallest k whose lower tail exceeds p).

21 timed repeats after one warm-up, median and max-min spread. Five of the six runs ran in parallel, so absolute times share a loaded CPU; speedups are within-run. numpy 2.5.4, Python 3.13.5, no network, nothing bound.

Commands, from a scratch dir: `python3 bench.py`, `python3 bench.py 20261025_1855`, `python3 bench.py 777`, and the same three seeds with a second argument `shared`.

```python
import numpy as np, time, math, sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 20261026_255
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
res={n:tm(f) for n,f in [("floored MAD, sort row 3",lambda h,i:mad_sort(h)),("in-window beta-binomial, uint8",inwin),("leave-today-out NB, uint8 up",loo_up),("leave-today-out NB, uint8 near",loo_near),("LOO near, de-shared fit",loo_desh),("LOO near, de-shared + rescale",loo_desh_rs),("rescale, stored f16 means",loo_stored),("rescale, stored scalar",loo_scalar),("floored MAD, two-sided",lambda h,i:mad_2s(h)),("stored rescale, two-sided",loo_stored_2s)]}
base=res["floored MAD, sort row 3"][1]
print(f"numpy {np.__version__} seed {seed} shared {SHARED} blocks {NBLK} LUT {t_build*1e3:.0f} ms LOO build {t_build2*1e3:.0f} ms LOO table {LOO.nbytes} B stored {g8.nbytes+n8.nbytes+d8.nbytes} B")
for n,(v,med,spr) in res.items():
    vs=set(v.tolist()); fa=len(vs-planted-quiet); caught=len(planted&vs); cq=len(quiet&vs)
    print(f"{n:32s} {med:7.2f} ms (spread {spr:5.2f})  {base/med:5.2f}x vs MAD  alarms {len(v):5d}  false {fa:5d}  caught {caught}/{len(planted)}  quiet {cq}/{len(quiet)}")
print(f"stored means {sum(m.nbytes for m in MS)} B, lo table {LOOLO.nbytes} B")
print("agree stored-vs-recomputed rescale:", set(res['rescale, stored f16 means'][0].tolist())==set(res['LOO near, de-shared + rescale'][0].tolist()))
```

## Results

Timings in ms, median (spread over 21 repeats). Speedup is against one-sided floored MAD in the same run. "Hot" is planted hot blocks caught, "quiet" is planted quiet blocks caught.

| run | floored MAD (one-sided) | recomputed rescale (last run) | stored f16 means | stored scalar | MAD two-sided | stored rescale two-sided |
|---|---|---|---|---|---|---|
| 0255 | 36.32 (1.43), 2,094 false, 6/6 hot | 5.45 (1.18), 6.67x, 0 false, 6/6 | 4.12 (0.53), 8.81x, 0 false, 6/6 | 4.77 (0.65), 7.62x, 0 false | 36.99 (2.58), 2,485 false, 4/6 quiet | 4.93 (1.42), 7.37x, 0 false, 6/6 hot, 6/6 quiet |
| 1855 | 36.36 (4.25), 1,993 false, 6/6 | 5.36 (2.01), 6.78x, 0 false, 6/6 | 4.11 (0.75), 8.86x, 0 false, 6/6 | 4.82 (0.67), 7.55x, 0 false | 37.71 (6.90), 2,434 false, 5/6 quiet | 4.73 (0.67), 7.69x, 0 false, 6/6, 6/6 |
| 777 | 36.97 (5.74), 2,075 false, 6/6 | 5.54 (0.70), 6.67x, 0 false, 6/6 | 4.14 (2.92), 8.93x, 0 false, 6/6 | 4.81 (0.67), 7.68x, 0 false | 37.41 (4.71), 2,529 false, 3/6 quiet | 5.14 (1.82), 7.19x, 0 false, 6/6, 5/6 |
| 0255 shared | 36.95 (1.55), 3,696 false, 6/6 | 5.37 (2.49), 6.88x, 1 false, 6/6 | 4.12 (0.43), 8.98x, 1 false, 6/6 | 4.78 (0.68), 7.73x, 1 false | 37.87 (6.38), 4,319 false, 1/6 quiet | 4.73 (0.67), 7.82x, 1 false, 6/6, 5/6 |
| 1855 shared | 36.29 (3.02), 4,043 false, 6/6 | 5.62 (1.46), 6.46x, 0 false, 6/6 | 4.20 (3.77), 8.64x, 0 false, 6/6 | 4.87 (0.64), 7.45x, 0 false | 37.50 (3.37), 4,410 false, 4/6 quiet | 4.72 (0.55), 7.69x, 0 false, 6/6, 6/6 |
| 777 shared | 37.09 (2.19), 5,534 false, 6/6 | 5.46 (0.49), 6.79x, 0 false, 6/6 | 4.12 (2.03), 9.00x, 0 false, 6/6 | 5.44 (1.35), 6.82x, 0 false | 38.85 (6.12), 6,063 false, 2/6 quiet | 4.74 (1.27), 7.82x, 0 false, 6/6, 6/6 |
| **total** | **19,435 false, 36/36 hot** | **1 false, 36/36** | **1 false, 36/36** | **1 false, 36/36** | **22,240 false, 19/36 quiet** | **1 false, 36/36 hot, 34/36 quiet** |

Stored float16 means give exactly the same alarm sets as recomputing the means in all six runs, and cut the rescaled path from 5.36 to 5.62 ms down to 4.11 to 4.20 ms (about 1.3x, so it does not get back to 3 ms: the 8-wide dot per block is the remaining cost). The single stored scalar is slower than the per-day version (4.77 to 5.44 ms), because summing the history rows costs about as much as the dot. Last run's cheap nearest-grid path (no rescale) still loses hot plants and gains false alarms in the shared runs here (34 of 36 hot, 6 false), so the rescale is still worth its cost.

The two-sided version costs 0.6 to 1.0 ms more than one-sided (one extra 278,596-byte table and one more comparison per block) and is 7.2 to 7.8x faster than a two-sided floored MAD. It catches 34 of 36 quiet blocks against MAD's 19 of 36, at 1 false alarm in about 3.05 million block-days against MAD's 22,240. MAD misses quiet blocks because its floor (5 times the square root of the median) is wider than the drop on modest-rate tensors.

## Verdict

**Viable.** The two-sided stored-means leave-today-out alarm is 7.2 to 7.8x faster than two-sided floored MAD, catches every hot plant (36 of 36) and 34 of 36 quiet plants (MAD 19 of 36), with 1 false alarm against 22,240. Storing the day means alone is a real but smaller gain (1.3x over recomputing, under the 1.5x bar on its own) for 2,940 bytes, and it never changes an alarm. Synthetic data only.

## Pair files this wave

- No pair report reached this run. JuniorLLM's tip at clone time was [c5a1b99](https://github.com/cloudcover95/JuniorLLM/commit/c5a1b99) (cloudcover95, 02:50 MT), which adds rails/additive/additive_guard.py (an offline additive-only diff guard); it carries no roadmap item in its message, so it is listed here for the ledger rather than claimed as this wave's pair file. This builds on pair A's 2026-10-24 05:05 files: [JuniorLLM scripts/daily_audit_1645_i2s_trit_flip_count_key.py @ 8a1b9fa](https://github.com/cloudcover95/JuniorLLM/blob/8a1b9fa/scripts/daily_audit_1645_i2s_trit_flip_count_key.py) (Trit Pack) and [JuniorQuant src/daily_audit_1645_f32_gamma_receipt.py @ 55ac5fb](https://github.com/cloudcover95/JuniorQuant/blob/55ac5fb/src/daily_audit_1645_f32_gamma_receipt.py) (Winsor).
- Paths honored: rails/additive/ (c5a1b99) and the JuniorHome paths flagged last wave are untouched; this note touches only JuniorLLM junior_osai/vault/findings/.

Previous: [[2026-10-25_1855_daily-audit-deshared-phi-rescaled-loo-vs-floored-mad]]

## Next question

The remaining cost is the 8-wide float dot per block. Can the rescale be folded into the lookup instead (index a per-tensor cutoff table by the unscaled history total and shift the cutoff by today's scale factor, or bucket the scale factor into a few uint8 steps), so the two-sided alarm runs near the 3 ms one-sided lookup with the same 36 of 36 hot and 34 of 36 quiet?

close: b34daa6b70a73d18 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-25_1855)
