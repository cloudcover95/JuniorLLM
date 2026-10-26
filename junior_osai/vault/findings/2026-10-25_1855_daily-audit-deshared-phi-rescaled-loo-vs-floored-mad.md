---
local: true
date: 2026-10-25
wave: "2026-10-25 18:05 (bench 18:55, ran 18:59)"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: 16:45 daily audit (alarm on hot trit-flip blocks)"
verdict: viable
tags: [ternary, i2s, daily-audit, trit-flips, overdispersion, negative-binomial, leave-today-out, shared-day-factor, lookup-table, recall, bench]
---

# Daily audit: de-shared phi fit plus history rescale for the leave-today-out trit-flip alarm (BitNet 2B4T shapes)

## Question

Last run ([[2026-10-25_1455_daily-audit-leave-today-out-nb-vs-in-window-beta-binomial]]) found nearest-grid rounding with the leave-today-out negative binomial got 34 of 36 planted hot blocks at 2 false alarms, and blamed the remaining misses on the stored per-tensor phi absorbing the tensor-wide day factor. If the stored phi is fitted after dividing each history day by its tensor-wide mean, with nearest-grid rounding and the leave-today-out test, does recall reach 6 of 6 at 0 false alarms in all six runs while staying at least 1.5x faster than floored MAD? And does it also need today's test itself to be rescaled for the shared day factor?

## Method

Same synthetic whole model and generator as the last two notes (seeds 20261025_1055 and 777 reproduce their draws, 20261025_1855 is new): 210 ternary tensors of BitNet b1.58 2B4T, 508,800 blocks of 4,096 trits, int32 per-block flip histograms from the I2_S XOR+popcount path, 9 days, per-tensor Gamma day factor (phi log-uniform 0.005 to 0.1), optional tensor-wide shared day factor, 6 planted hot blocks on the last day. The phi fits run on yesterday's 7 history days and are stored as uint8 grid indexes (untimed, 210 bytes each).

Six per-tensor-loop paths at p = 1e-7. Paths 1 to 4 are last run's (floored MAD, in-window beta-binomial rounded up, leave-today-out rounded up, leave-today-out nearest). New:

5. Leave-today-out nearest with the de-shared fit: each history day divided by its tensor mean over the 7-day mean before the moment fit.
6. Same as 5, plus each history day rescaled to today's tensor-wide level before the sum (S = round(sum over d of h_d times mean_today / mean_d)). This adds one mean per row and one 8-wide dot per block, all inside the timed loop.

Same 17 x 4,097 int32 tables (278,596 bytes each). 21 timed repeats after one warm-up, median and max-min spread. The three seeds ran in parallel (two runs per seed in sequence), so absolute times share a loaded CPU; speedups are within-run. numpy 2.5.4, Python 3.13.5, no network, nothing bound.

Commands, from a scratch dir: `python3 bench.py`, `python3 bench.py 20261025_1055`, `python3 bench.py 777`, and the same three seeds with a second argument `shared`.

```python
import numpy as np, time, math, sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 20261025_1855
SHARED=len(sys.argv)>2 and sys.argv[2]=="shared"
rng=np.random.default_rng(seed); R=21; B=4096; D=9; P=1e-7
H_, KV, FF, L = 2560, 640, 6912, 30
shapes=[("q",H_,H_),("k",H_,KV),("v",H_,KV),("o",H_,H_),("gate",H_,FF),("up",H_,FF),("down",FF,H_)]
tensors=[(f"L{l}.{n}",a*b//B) for l in range(L) for n,a,b in shapes]
# 9 days: days 0..7 = yesterday's window (stored phi fitted from its 7 history days 0..6), days 1..8 = today's window
hs=[]; phis=[]; planted=set(); off=0
for i,(name,nb) in enumerate(tensors):
    lam=float(np.exp(rng.uniform(np.log(2),np.log(400))))
    phi=float(np.exp(rng.uniform(np.log(0.005),np.log(0.1)))); a=1/phi
    G=rng.gamma(a,1/a,size=(D,nb))
    if SHARED:  # tensor-wide day factor, phi_s log-uniform 0.005..0.05
        ps=float(np.exp(rng.uniform(np.log(0.005),np.log(0.05)))); G=G*rng.gamma(1/ps,ps,size=(D,1))
    h=rng.poisson(lam*G).astype(np.int32)
    if i%37==5:
        j=int(rng.integers(nb)); sd=math.sqrt(lam+lam*lam*phi); h[-1,j]+=int(10*sd+20); planted.add(off+j)
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
t0=time.perf_counter(); LOO=np.stack([build_loo(float(g)) for g in GRID]); t_build2=time.perf_counter()-t0
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
def run(f):
    out=[];o=0
    for i,h in enumerate(W): out.append(np.flatnonzero(f(h,i))+o); o+=h.shape[1]
    return np.concatenate(out)
def tm(f):
    v=run(f); ts=[]
    for _ in range(R):
        s=time.perf_counter(); run(f); ts.append(time.perf_counter()-s)
    ts=np.array(ts)*1e3; return v,float(np.median(ts)),float(ts.max()-ts.min())
res={n:tm(f) for n,f in [("floored MAD, sort row 3",lambda h,i:mad_sort(h)),("in-window beta-binomial, uint8",inwin),("leave-today-out NB, uint8 up",loo_up),("leave-today-out NB, uint8 near",loo_near),("LOO near, de-shared fit",loo_desh),("LOO near, de-shared + rescale",loo_desh_rs)]}
base=res["floored MAD, sort row 3"][1]
print(f"numpy {np.__version__} seed {seed} shared {SHARED} blocks {NBLK} LUT {t_build*1e3:.0f} ms LOO build {t_build2*1e3:.0f} ms LOO table {LOO.nbytes} B stored {g8.nbytes+n8.nbytes+d8.nbytes} B")
for n,(v,med,spr) in res.items():
    fa=len(set(v.tolist())-planted); caught=len(planted&set(v.tolist()))
    print(f"{n:32s} {med:7.2f} ms (spread {spr:5.2f})  {base/med:5.2f}x vs MAD  alarms {len(v):5d}  false {fa:5d}  caught {caught}/{len(planted)}")
o=0; vl=set(res["LOO near, de-shared + rescale"][0].tolist())
for i,h in enumerate(W):
    nb=h.shape[1]
    for p in sorted(planted):
        if o<=p<o+nb:
            j=p-o; S=int(h[:-1,j].sum()); print(f"plant t{i} phi {phis[i]:.3f} g {GRID[g8[i]]:.3f} gD {GRID[d8[i]]:.3f} today {h[-1,j]} S {S} inwin_cut {LUT[g8[i]][min(S+h[-1,j],CAP)]} loo_cut {LOO[d8[i]][min(S,CAP)]} {'hit' if p in vl else 'MISS'}")
    o+=nb
```

## Results

Timings in ms, median (spread over 21 repeats). Speedup is against floored MAD in the same run.

| run | floored MAD | LOO nearest (last run's best) | LOO nearest, de-shared fit | LOO nearest, de-shared + rescale |
|---|---|---|---|---|
| 1855 | 36.36 (2.85), 2,037 false, 6/6 | 3.00 (1.22), 12.1x, 1 false, 6/6 | 2.90 (0.89), 12.5x, 1 false, 6/6 | 5.32 (0.48), 6.84x, 1 false, 6/6 |
| 1055 | 36.55 (2.98), 2,104 false, 6/6 | 2.90 (0.43), 12.6x, 1 false, 5/6 | 2.92 (1.29), 12.5x, 1 false, 5/6 | 5.43 (0.58), 6.73x, 0 false, 5/6 |
| 777 | 36.60 (4.47), 2,395 false, 6/6 | 2.87 (0.50), 12.7x, 0 false, 6/6 | 2.92 (1.72), 12.6x, 0 false, 6/6 | 5.41 (0.37), 6.77x, 0 false, 6/6 |
| 1855 shared | 37.04 (7.66), 3,507 false, 6/6 | 2.99 (0.50), 12.4x, 1 false, 6/6 | 3.08 (0.52), 12.0x, 11 false, 6/6 | 5.56 (1.50), 6.66x, 0 false, 6/6 |
| 1055 shared | 36.81 (2.63), 3,804 false, 6/6 | 2.88 (0.40), 12.8x, 0 false, 6/6 | 2.89 (0.33), 12.7x, 23 false, 6/6 | 5.37 (0.87), 6.86x, 0 false, 6/6 |
| 777 shared | 37.03 (4.44), 3,765 false, 5/6 | 2.98 (0.70), 12.4x, 1 false, 5/6 | 2.94 (0.35), 12.6x, 87 false, 6/6 | 5.35 (0.64), 6.92x, 0 false, 6/6 |
| **total** | **17,612 false, 35/36** | **4 false, 34/36** | **123 false, 35/36** | **1 false, 35/36** |

The de-shared fit fixes the stored phi (for example true 0.007 now reads 0.008 instead of 0.116, and true 0.009 reads 0.030 instead of 0.059), but used alone it is worse in the shared runs: 11 to 87 false alarms, because the smaller phi gives a tighter cutoff while today's shared day factor still moves every block in the tensor. Rescaling the history days to today's level fixes that and gives 0 false alarms in all three shared runs. The one remaining miss (seed 1055, phi 0.060, today 410 against cutoff 433 from history total 979) is a plant smaller than the day-factor noise at p = 1e-7, and the one remaining false alarm (seed 1855, unshared) is flagged by every path, including MAD, so it looks like a natural extreme block. The rescale costs about 2.5 ms (one row mean and one 8-wide dot per block), so path 6 is 1.85x slower than path 4.

## Verdict

**Viable.** De-shared fit plus history rescale is the first ternary-histogram path that matches floored MAD's recall (35 of 36 each) while staying 6.7 to 6.9x faster, at 1 false alarm in about 3.05 million block-days against MAD's 17,612. The asked bar (6 of 6 at 0 false in all six runs) is not quite met, because of one plant that no path but MAD catches. If the 1.85x extra cost over last run's 12x path matters more than the shared-factor false alarms, last run's nearest-grid path (34 of 36, 4 false) is the cheaper pick. Synthetic data only.

## Pair files this wave

- None. Quant Chief's 18:05 run (00:12Z) gave both pairs "no gap" again (ninth idle wave in a row; roadmap blobs unchanged), and JuniorLLM's tip was still my own 5a6cbae. Builds on pair A's 2026-10-24 05:05 files: [JuniorLLM scripts/daily_audit_1645_i2s_trit_flip_count_key.py @ 8a1b9fa](https://github.com/cloudcover95/JuniorLLM/blob/8a1b9fa/scripts/daily_audit_1645_i2s_trit_flip_count_key.py) (Trit Pack) and [JuniorQuant src/daily_audit_1645_f32_gamma_receipt.py @ 55ac5fb](https://github.com/cloudcover95/JuniorQuant/blob/55ac5fb/src/daily_audit_1645_f32_gamma_receipt.py) (Winsor).
- Paths honored: the Chief's flagged JuniorHome d8a0398 (scripts/advance_t13_named.py, T13+), plus new JuniorHome 49903b8 and 0eba850 (18:21 MT), are untouched; this note touches only JuniorLLM junior_osai/vault/findings/.

Previous: [[2026-10-25_1455_daily-audit-leave-today-out-nb-vs-in-window-beta-binomial]]

## Next question

Can the per-day tensor means for the rescale be stored with the histograms (8 float16 per tensor, 3,360 bytes for the whole model window) so the rescaled leave-today-out path drops back near 3 ms, keeping 35 of 36 at 1 false alarm, and does a two-sided version (also alarming on blocks that suddenly go quiet) cost anything extra?

close: 8289ab1c15cf5d65 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-25_1455)
