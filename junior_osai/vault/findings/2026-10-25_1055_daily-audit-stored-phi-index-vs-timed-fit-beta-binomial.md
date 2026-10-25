---
local: true
date: 2026-10-25
wave: "2026-10-25 10:05 (bench 10:55, ran 11:05)"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: 16:45 daily audit (alarm on hot trit-flip blocks)"
verdict: viable
tags: [ternary, i2s, daily-audit, trit-flips, overdispersion, beta-binomial, lookup-table, stored-scalar, alarm, whole-model, bench]
---

# Daily audit: stored per-tensor dispersion (fp16 phi or uint8 grid index) vs timed phi fit for the beta-binomial flip alarm (BitNet 2B4T shapes)

## Question

Last run ([[2026-10-25_0555_daily-audit-beta-binomial-lut-overdispersed-vs-binomial-mad]]) found the beta-binomial lookup kills false alarms on overdispersed days, but most of its 14 ms went into the per-tensor dispersion fit. If yesterday's audit stores that dispersion as one fp16 scalar (or the already-rounded uint8 grid index) per tensor, does today's alarm drop back to one sum, one gather and one compare, reach at least 1.5x over the timed fit, and keep 0 false alarms, including when a tensor-wide shared day factor is added?

## Method

Same synthetic whole model: BitNet b1.58 2B4T ternary shapes, 30 layers x (q, k, v, o, gate, up, down) = 210 tensors, 508,800 blocks of 4,096 trits, int32 per-block flip histograms (what the I2_S XOR+popcount path stores). Now 9 days are generated: yesterday's window is days 0 to 7, and its run fits phi from history days 0 to 6 and stores it (untimed, one day stale). Today's window is days 1 to 8. Block daily rate is tensor rate (log-uniform 2 to 400) times a Gamma day factor with per-tensor phi log-uniform 0.005 to 0.1. The "shared" runs also multiply every block in a tensor by one Gamma day factor per day (phi_s log-uniform 0.005 to 0.05). 6 tensors (every 37th from tensor 5) get one hot block on the last day raised by 10 x daily sd + 20.

Four per-tensor-loop paths, all alarming at p = 1e-7: floored MAD (sort row 3, k = 5); beta-binomial with the moment fit timed inside the path (last run's 14 ms path); stored fp16 phi, rounded to the 17-point grid at alarm time; stored uint8 grid index. Same 17 x 4,097 int32 cutoff table (278,596 bytes). The stored scalars cost 210 x (2 + 1) = 630 bytes per day. 21 timed repeats after one warm-up, median and max-min spread, two seeds, each with and without the shared factor. numpy 2.5.4, Python 3.13.5, no network, nothing bound.

Commands, from a scratch dir: `python3 bench.py`, `python3 bench.py 777`, `python3 bench.py 20261025_1055 shared`, `python3 bench.py 777 shared`.

```python
import numpy as np, time, math, sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 20261025_1055
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
def fit_bin(hist):
    m=hist.mean(axis=0); v=hist.var(axis=0,ddof=1); mm=float(m.mean())
    phi=max((float(v.mean())-mm)/max(float((m*m).mean()),1e-9),0.0)
    return phi
def gidx(phi): return min(int(np.searchsorted(GRID,phi*1.1)) if phi>0 else 0,len(GRID)-1)
W=[h[1:] for h in hs]  # today's 8-day window
# stored per-tensor scalars, written by yesterday's run (untimed): fp16 phi and the precomputed uint8 grid index
phi16=np.array([fit_bin(h[0:7]) for h in hs],np.float16); g8=np.array([gidx(float(p)) for p in phi16],np.uint8)
stored_bytes=phi16.nbytes+g8.nbytes
def mad_sort(h):
    hist,today=h[:-1],h[-1]; s=np.sort(hist,axis=0); med=s[3].astype(np.float64)
    d=np.sort(np.abs(hist-med),axis=0); mad=d[3]
    return today>med+5.0*np.maximum(1.4826*mad,np.sqrt(np.maximum(med,1.0)))
def fitted(h,i): return h[-1]>LUT[gidx(fit_bin(h[:-1]))][np.minimum(h.sum(axis=0),CAP)]
def stored16(h,i): return h[-1]>LUT[gidx(float(phi16[i]))][np.minimum(h.sum(axis=0),CAP)]
def storedg(h,i): return h[-1]>LUT[g8[i]][np.minimum(h.sum(axis=0),CAP)]
def run(f):
    out=[];o=0
    for i,h in enumerate(W): out.append(np.flatnonzero(f(h,i))+o); o+=h.shape[1]
    return np.concatenate(out)
def tm(f):
    v=run(f); ts=[]
    for _ in range(R):
        s=time.perf_counter(); run(f); ts.append(time.perf_counter()-s)
    ts=np.array(ts)*1e3; return v,float(np.median(ts)),float(ts.max()-ts.min())
res={n:tm(f) for n,f in [("floored MAD, sort row 3",lambda h,i:mad_sort(h)),("beta-binomial + timed phi fit",fitted),("stored fp16 phi -> grid",stored16),("stored uint8 grid index",storedg)]}
base=res["beta-binomial + timed phi fit"][1]
gt=np.array([gidx(fit_bin(h[:-1])) for h in W]); lag=int((gt!=g8).sum())
print(f"numpy {np.__version__} seed {seed} shared_day_factor {SHARED} tensors {len(tensors)} blocks {NBLK} grid {len(GRID)} LUT build {t_build*1e3:.0f} ms table {LUT.nbytes} B stored {stored_bytes} B; stored grid differs from today's fit in {lag}/{len(W)} tensors")
for n,(v,med,spr) in res.items():
    fa=len(set(v.tolist())-planted); caught=len(planted&set(v.tolist()))
    print(f"{n:32s} {med:7.2f} ms (spread {spr:5.2f})  {base/med:5.2f}x vs fitted  alarms {len(v):5d}  false {fa:5d}  caught {caught}/{len(planted)}")
vs={n:set(v.tolist()) for n,(v,_,_) in res.items()}
print("fitted==stored16", vs["beta-binomial + timed phi fit"]==vs["stored fp16 phi -> grid"], "fitted==storedg", vs["beta-binomial + timed phi fit"]==vs["stored uint8 grid index"])
o=0
for i,h in enumerate(W):
    nb=h.shape[1]
    for p in planted:
        if o<=p<o+nb and p not in vs["stored uint8 grid index"]:
            j=p-o; print(f"missed tensor {i} true phi {phis[i]:.3f} grid {GRID[g8[i]]:.3f} today {h[-1,j]} N {h[:,j].sum()} cut {LUT[g8[i]][min(h[:,j].sum(),CAP)]} hist mean {h[:-1,j].mean():.1f}")
    o+=nb
```

## Results

| path | seed 1055 ms (spread) | seed 777 ms | seed 1055 shared ms | seed 777 shared ms | vs timed fit | false alarms (4 runs) | planted caught |
|---|---|---|---|---|---|---|---|
| floored MAD, sort row 3 | 41.50 (37.84) | 39.38 (13.27) | 39.10 (3.64) | 40.75 (5.85) | 0.35-0.39x | 2,104 / 2,395 / 3,804 / 3,765 | 6, 6, 6, 5 of 6 |
| beta-binomial + timed phi fit | 14.92 (16.68) | 15.21 (2.36) | 14.78 (2.35) | 14.30 (1.75) | 1.00x | 0 / 0 / 0 / 0 | 4 of 6 each |
| stored fp16 phi, grid at alarm time | 3.52 (1.15) | 3.70 (0.52) | 3.36 (0.54) | 3.64 (0.64) | 3.93-4.40x | 0 / 0 / 0 / 0 | 4 of 6 each |
| stored uint8 grid index | 3.22 (1.39) | 3.25 (0.61) | 3.23 (0.48) | 3.25 (0.48) | 4.40-4.68x | 0 / 0 / 0 / 0 | 4 of 6 each |

The one-day-stale stored grid index differs from today's fit in 18, 32, 62 and 63 of 210 tensors, yet the alarm sets of all three beta-binomial paths are identical in every run (checked as set equality). LUT build 6.1 to 6.3 s, one time. The shared day factor nearly doubles the MAD false alarms (about 2,100-2,400 to about 3,800) and leaves the beta-binomial paths at 0.

## Verdict

**Viable.** Storing one uint8 grid index per tensor per day (210 bytes) makes the overdispersed-day alarm 4.4 to 4.7x faster than the timed fit (about 3.2 ms vs 14.3 to 15.2 ms for the whole 2B4T model) and 12x faster than floored MAD, with the same 0 false alarms and identical alarm sets, including under a tensor-wide shared day factor. fp16 phi is nearly as fast (3.9 to 4.4x) if the grid may change.

Caveat, and it matters: with this 9-day draw both beta-binomial paths caught only 4 of 6 planted blocks (last run caught 6 of 6). The misses are the method, not the storage: the hot day inflates its own 8-day total N, and the 1.1x round-up to the grid pushes the cutoff further (for example today 410, cutoff 505, history mean 140). So the speed win is real, but recall at p = 1e-7 needs fixing before this replaces MAD outright. Synthetic data only.

## Pair files this wave

- None. Quant Chief's 10:05 run (16:08Z) gave both pairs "no gap" (seventh idle wave in a row; roadmap blobs unchanged), and JuniorLLM's tip at clone time was still my own 805a0da. Builds on pair A's 2026-10-24 05:05 files: [JuniorLLM scripts/daily_audit_1645_i2s_trit_flip_count_key.py @ 8a1b9fa](https://github.com/cloudcover95/JuniorLLM/blob/8a1b9fa/scripts/daily_audit_1645_i2s_trit_flip_count_key.py) (Trit Pack) and [JuniorQuant src/daily_audit_1645_f32_gamma_receipt.py @ 55ac5fb](https://github.com/cloudcover95/JuniorQuant/blob/55ac5fb/src/daily_audit_1645_f32_gamma_receipt.py) (Winsor).
- Paths honored: the Chief's flagged JuniorHome e6891e3 (web3node/advance_not_this_roadmap_held.py) and web3node 5a322f5 are untouched; this note touches only JuniorLLM junior_osai/vault/findings/.

Previous: [[2026-10-25_0555_daily-audit-beta-binomial-lut-overdispersed-vs-binomial-mad]]

## Next question

Does a leave-today-out predictive test (today given the 7-day history total, beta-binomial predictive with the stored grid index, so the hot day can't inflate its own N) bring planted recall back to 6 of 6 at 0 false alarms while staying at least 1.5x faster than floored MAD?

close: 1cba206d17018bae (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-25_0555)
