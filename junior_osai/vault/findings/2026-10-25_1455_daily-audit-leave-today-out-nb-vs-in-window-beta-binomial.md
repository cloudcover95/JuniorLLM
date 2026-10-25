---
local: true
date: 2026-10-25
wave: "2026-10-25 14:05 (bench 14:55, ran 15:00)"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: 16:45 daily audit (alarm on hot trit-flip blocks)"
verdict: inconclusive
tags: [ternary, i2s, daily-audit, trit-flips, overdispersion, beta-binomial, negative-binomial, leave-today-out, lookup-table, recall, bench]
---

# Daily audit: leave-today-out negative-binomial predictive vs in-window beta-binomial for the trit-flip alarm (BitNet 2B4T shapes)

## Question

Last run ([[2026-10-25_1055_daily-audit-stored-phi-index-vs-timed-fit-beta-binomial]]) made the overdispersed-day alarm 4.4 to 4.7x faster by storing a uint8 grid index per tensor, but both beta-binomial paths caught only 4 of 6 planted hot blocks, because the hot day inflates its own 8-day total N. If today is tested against a predictive built from the 7 history days only (leave-today-out), does planted recall come back to 6 of 6 at 0 false alarms while staying at least 1.5x faster than floored MAD?

## Method

Same synthetic whole model and data generator as last run (the 20261025_1055 and 777 seeds reproduce last run's draws exactly, and 20261025_1455 is new): 210 ternary tensors of BitNet b1.58 2B4T, 508,800 blocks of 4,096 trits, int32 per-block flip histograms from the I2_S XOR+popcount path, 9 days, per-tensor Gamma day factor (phi log-uniform 0.005 to 0.1), optional tensor-wide shared day factor, 6 planted hot blocks on the last day. Yesterday's run stores the per-tensor phi as a uint8 grid index (untimed).

Four per-tensor-loop paths, all at p = 1e-7:

1. Floored MAD (sort row 3, k = 5), the baseline for the 1.5x bar.
2. In-window beta-binomial with the stored uint8 index rounded up 1.1x to the 17-point grid (last run's 3.2 ms path).
3. Leave-today-out: today X against a negative binomial moment-matched to the 7-day total S, mean S/7 and variance (mu + phi mu^2)(1 + 1/7), same rounded-up index. One sum over 7 rows, one gather, one compare.
4. Same as 3 but with the stored index rounded to the nearest grid point instead of up.

Both cutoff tables are 17 x 4,097 int32 (278,596 bytes each). Stored scalars: 210 bytes per index. 21 timed repeats after one warm-up, median and max-min spread, three seeds, each with and without the shared factor. numpy 2.5.4, Python 3.13.5, no network, nothing bound.

Commands, from a scratch dir: `python3 bench.py`, `python3 bench.py 20261025_1055`, `python3 bench.py 777`, and the same three with a second argument `shared`.

```python
import numpy as np, time, math, sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 20261025_1455
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
phi16=np.array([fit_bin(h[0:7]) for h in hs],np.float16)
g8=np.array([gidx(float(p)) for p in phi16],np.uint8); n8=np.array([gnear(float(p)) for p in phi16],np.uint8)
def mad_sort(h):
    hist,today=h[:-1],h[-1]; s=np.sort(hist,axis=0); med=s[3].astype(np.float64)
    d=np.sort(np.abs(hist-med),axis=0); mad=d[3]
    return today>med+5.0*np.maximum(1.4826*mad,np.sqrt(np.maximum(med,1.0)))
def inwin(h,i): return h[-1]>LUT[g8[i]][np.minimum(h.sum(axis=0),CAP)]
def loo_up(h,i): return h[-1]>LOO[g8[i]][np.minimum(h[:-1].sum(axis=0),CAP)]
def loo_near(h,i): return h[-1]>LOO[n8[i]][np.minimum(h[:-1].sum(axis=0),CAP)]
def run(f):
    out=[];o=0
    for i,h in enumerate(W): out.append(np.flatnonzero(f(h,i))+o); o+=h.shape[1]
    return np.concatenate(out)
def tm(f):
    v=run(f); ts=[]
    for _ in range(R):
        s=time.perf_counter(); run(f); ts.append(time.perf_counter()-s)
    ts=np.array(ts)*1e3; return v,float(np.median(ts)),float(ts.max()-ts.min())
res={n:tm(f) for n,f in [("floored MAD, sort row 3",lambda h,i:mad_sort(h)),("in-window beta-binomial, uint8",inwin),("leave-today-out NB, uint8 up",loo_up),("leave-today-out NB, uint8 near",loo_near)]}
base=res["floored MAD, sort row 3"][1]
print(f"numpy {np.__version__} seed {seed} shared {SHARED} blocks {NBLK} LUT {t_build*1e3:.0f} ms LOO build {t_build2*1e3:.0f} ms LOO table {LOO.nbytes} B stored {g8.nbytes+n8.nbytes} B")
for n,(v,med,spr) in res.items():
    fa=len(set(v.tolist())-planted); caught=len(planted&set(v.tolist()))
    print(f"{n:32s} {med:7.2f} ms (spread {spr:5.2f})  {base/med:5.2f}x vs MAD  alarms {len(v):5d}  false {fa:5d}  caught {caught}/{len(planted)}")
o=0; vl=set(res["leave-today-out NB, uint8 up"][0].tolist())
for i,h in enumerate(W):
    nb=h.shape[1]
    for p in sorted(planted):
        if o<=p<o+nb:
            j=p-o; S=int(h[:-1,j].sum()); print(f"plant t{i} phi {phis[i]:.3f} g {GRID[g8[i]]:.3f} today {h[-1,j]} S {S} inwin_cut {LUT[g8[i]][min(S+h[-1,j],CAP)]} loo_cut {LOO[g8[i]][min(S,CAP)]} {'hit' if p in vl else 'MISS'}")
    o+=nb
```

## Results

Timings in ms, median (spread over 21 repeats). Speedup is against floored MAD in the same run. False alarms and planted catches are per run.

| run | floored MAD | in-window BB, up | leave-today-out, up | leave-today-out, nearest |
|---|---|---|---|---|
| 1455 | 38.41 (3.97), 2,007 false, 6/6 | 3.18 (1.76), 12.1x, 0 false, 6/6 | 3.07 (0.31), 12.5x, 0 false, 6/6 | 3.08 (0.47), 12.5x, 0 false, 6/6 |
| 1055 | 39.75 (16.92), 2,104 false, 6/6 | 3.45 (1.99), 11.5x, 0 false, 4/6 | 3.31 (0.50), 12.0x, 0 false, 5/6 | 3.33 (0.54), 12.0x, 1 false, 5/6 |
| 777 | 37.83 (8.05), 2,395 false, 6/6 | 3.57 (4.27), 10.6x, 0 false, 4/6 | 3.23 (0.57), 11.7x, 0 false, 5/6 | 3.18 (0.51), 11.9x, 0 false, 6/6 |
| 1455 shared | 37.27 (15.81), 3,994 false, 6/6 | 3.12 (1.14), 12.0x, 0 false, 5/6 | 2.98 (0.81), 12.5x, 0 false, 5/6 | 2.97 (0.25), 12.6x, 0 false, 6/6 |
| 1055 shared | 38.55 (8.25), 3,804 false, 6/6 | 3.22 (1.70), 12.0x, 0 false, 4/6 | 3.15 (8.00), 12.3x, 0 false, 5/6 | 3.00 (0.66), 12.8x, 0 false, 6/6 |
| 777 shared | 37.52 (1.92), 3,765 false, 5/6 | 3.31 (2.30), 11.3x, 0 false, 4/6 | 3.08 (0.52), 12.2x, 0 false, 4/6 | 3.01 (0.36), 12.5x, 1 false, 5/6 |
| **total** | **18,069 false, 35/36** | **0 false, 27/36** | **0 false, 30/36** | **2 false, 34/36** |

The leave-today-out cutoff is only 1 to 6% below the in-window cutoff on the missed blocks (for example today 410, history total S 979, in-window cut 505, leave-today-out cut 502, grid 0.083 vs true phi 0.060). In one shared run the stored fit read phi as 0.116 for a tensor whose true phi is 0.007, because the pooled per-tensor fit absorbs the tensor-wide day factor, and there leave-today-out is no help at all (cut 1,294 vs today 763). The leave-today-out table took about 57 s to build (one time, slow only because of the scalar lgamma loop), against about 6 s for the in-window table.

## Verdict

**Inconclusive.** Leave-today-out by itself does not restore recall: 30 of 36 vs 27 of 36, still 0 false alarms, and the speed is unchanged (11.7 to 12.5x faster than floored MAD, about 1.03 to 1.1x faster than in-window because it sums 7 rows instead of 8). The bigger lever is the 1.1x round-up of the stored grid index: rounding to the nearest grid point gets 34 of 36 at 2 false alarms in about 3.05 million block-days, against MAD's 35 of 36 at 18,069 false alarms. So the 12x speed win from the earlier notes stands, but the question asked (6 of 6 at 0 false alarms) is not met by any path in every run. No new efficiency gain, so nothing to report. Synthetic data only.

## Pair files this wave

- None. Quant Chief's 14:05 run (20:12Z) gave both pairs "no gap" (eighth idle wave in a row; roadmap blobs unchanged), and JuniorLLM's tip at clone time was still my own df0e007. Builds on pair A's 2026-10-24 05:05 files: [JuniorLLM scripts/daily_audit_1645_i2s_trit_flip_count_key.py @ 8a1b9fa](https://github.com/cloudcover95/JuniorLLM/blob/8a1b9fa/scripts/daily_audit_1645_i2s_trit_flip_count_key.py) (Trit Pack) and [JuniorQuant src/daily_audit_1645_f32_gamma_receipt.py @ 55ac5fb](https://github.com/cloudcover95/JuniorQuant/blob/55ac5fb/src/daily_audit_1645_f32_gamma_receipt.py) (Winsor).
- Paths honored: the Chief's flagged JuniorHome 18aba3e (web3node/advance_paused_slots.py, non-additive, 14:12 MT) and e6891e3 are untouched; this note touches only JuniorLLM junior_osai/vault/findings/.

Previous: [[2026-10-25_1055_daily-audit-stored-phi-index-vs-timed-fit-beta-binomial]]

## Next question

Does fitting the stored phi per tensor after removing the tensor-wide day factor (divide each day by its tensor median before the moment fit), combined with nearest-grid rounding and the leave-today-out test, reach 6 of 6 planted at 0 false alarms in all six runs, while staying at least 1.5x faster than floored MAD?

close: d059efada2b09dd0 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-25_1055)
