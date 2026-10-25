---
local: true
date: 2026-10-25
wave: "2026-10-25 05:05 (bench 05:55, ran 06:05)"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: 16:45 daily audit (alarm on hot trit-flip blocks)"
verdict: viable
tags: [ternary, i2s, daily-audit, trit-flips, overdispersion, beta-binomial, lookup-table, alarm, whole-model, bench]
---

# Daily audit: beta-binomial lookup on overdispersed flip days vs plain binomial and floored median/MAD (BitNet 2B4T shapes)

## Question

Last run ([[2026-10-25_0255_daily-audit-binomial-tail-lut-vs-floored-mad]]) found a binomial "today given 8-day total" lookup ~15x faster than the floored MAD alarm with 0 false alarms, but only on stationary Poisson days. Real flip counts are likely overdispersed day to day. When they are, does a beta-binomial lookup (one dispersion value per tensor, moment-fitted from the 7 stored history days and rounded up to a 17-point grid) keep false alarms near zero where the plain binomial lookup breaks, and stay at least 1.5x faster than the floored MAD path?

## Method

Same synthetic whole model: BitNet b1.58 2B4T ternary shapes, 30 layers x (q, k, v, o, gate, up, down) = 210 tensors, 508,800 blocks of 4,096 trits, 8 days of int32 per-block flip histograms (what the I2_S XOR+popcount path stores). New: each block's daily rate is the tensor rate (log-uniform 2 to 400) times a Gamma(a, 1/a) day factor, with dispersion phi = 1/a log-uniform 0.005 to 0.1 per tensor (daily rate CV about 7% to 32%). Under that model today given the 8-day total N is exactly BetaBinomial(N, a, 7a), still rate-free. 6 tensors (every 37th from tensor 5) get one hot block on the last day raised by 10 x daily sd + 20.

Three per-tensor-loop alarm paths:

1. Floored MAD with the sort-row-3 median (k = 5), the baseline.
2. Plain binomial lookup from last run (cutoff[N], p = 1e-7).
3. Beta-binomial lookup: per tensor, phi_hat = (mean var - mean m) / mean m^2 from the 7 history days, times 1.1, rounded up to a grid of 0 plus 16 log-spaced values 0.002 to 0.32; alarm when today > LUT[grid, N]. The fit is inside the timed path.

Tables: 17 x 4,097 int32 (278,596 bytes), built once from exact log-pmf tails using integer lgamma tables plus three per-grid lgamma vectors, totals above 4,096 clamp to the last entry. 21 timed repeats after one warm-up, median and max-min spread, two seeds. numpy 2.5.4, Python 3.13.5, no network, nothing bound.

Commands, from a scratch dir: `python3 bench.py` and `python3 bench.py 777`.

```python
import numpy as np, time, math, sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 20261025_0555
rng=np.random.default_rng(seed); R=21; B=4096; D=8; P=1e-7
H_, KV, FF, L = 2560, 640, 6912, 30
shapes=[("q",H_,H_),("k",H_,KV),("v",H_,KV),("o",H_,H_),("gate",H_,FF),("up",H_,FF),("down",FF,H_)]
tensors=[(f"L{l}.{n}",a*b//B) for l in range(L) for n,a,b in shapes]
# overdispersed days: per block per day rate = lam*G, G~Gamma(a,1/a); phi=1/a log-uniform 0.005..0.1
hs=[]; phis=[]; planted=set(); off=0
for i,(name,nb) in enumerate(tensors):
    lam=float(np.exp(rng.uniform(np.log(2),np.log(400))))
    phi=float(np.exp(rng.uniform(np.log(0.005),np.log(0.1)))); a=1/phi
    G=rng.gamma(a,1/a,size=(D,nb))
    h=rng.poisson(lam*G).astype(np.int32)
    if i%37==5:
        j=int(rng.integers(nb)); sd=math.sqrt(lam+lam*lam*phi); h[-1,j]+=int(10*sd+20); planted.add(off+j)
    hs.append(h); phis.append(phi); off+=nb
NBLK=off; CAP=1<<12
lg=np.array([math.lgamma(i+1) for i in range(CAP+1)])
def cut_rows(logpmf):  # logpmf (n_N, CAP+1) with -inf beyond N; alarm if today > cut
    m=logpmf.max(axis=1,keepdims=True); pm=np.exp(logpmf-m); pm/=pm.sum(axis=1,keepdims=True)
    tail=np.cumsum(pm[:,::-1],axis=1)[:,::-1]           # P(X>=k)
    ok=tail<=P; first=np.where(ok.any(axis=1),ok.argmax(axis=1),CAP+1)
    return (first-1).astype(np.int32)
K=np.arange(CAP+1)
def build(phi):  # today | 8-day total N ~ BetaBinomial(N,a,7a); phi=0 -> Binomial(N,1/8)
    lut=np.empty(CAP+1,np.int32)
    if phi>0:
        a=1/phi; b=7*a
        LA=np.array([math.lgamma(k+a) for k in K]); LB=np.array([math.lgamma(k+b) for k in K])
        LAB=np.array([math.lgamma(n+a+b) for n in K]); c=math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)
    for s in range(0,CAP+1,256):
        N=K[s:s+256,None]; valid=K[None,:]<=N; NK=np.clip(N-K[None,:],0,CAP)
        if phi>0: lp=lg[N]-lg[K][None,:]-lg[NK]+LA[None,:]+LB[NK]-LAB[N]+c
        else:     lp=lg[N]-lg[K][None,:]-lg[NK]+K[None,:]*math.log(1/8)+NK*math.log(7/8)
        lut[s:s+256]=cut_rows(np.where(valid,lp,-np.inf))
    return lut
GRID=np.concatenate([[0.0],np.exp(np.linspace(np.log(0.002),np.log(0.32),16))])
t0=time.perf_counter(); LUT=np.stack([build(p) for p in GRID]); t_build=time.perf_counter()-t0
binlut=LUT[0]
def fit_bin(h):  # moment fit of phi from the 7 stored history days, round UP to the grid (conservative)
    hist=h[:-1]; m=hist.mean(axis=0); v=hist.var(axis=0,ddof=1)
    mm=float(m.mean()); phi=max((float(v.mean())-mm)/max(float((m*m).mean()),1e-9),0.0)
    return int(np.searchsorted(GRID,phi*1.1)) if phi>0 else 0
def mad_sort(h):
    hist,today=h[:-1],h[-1]; s=np.sort(hist,axis=0); med=s[3].astype(np.float64)
    d=np.sort(np.abs(hist-med),axis=0); mad=d[3]
    return today>med+5.0*np.maximum(1.4826*mad,np.sqrt(np.maximum(med,1.0)))
def binom(h): return h[-1]>binlut[np.minimum(h.sum(axis=0),CAP)]
def betabin(h):
    return h[-1]>LUT[min(fit_bin(h),len(GRID)-1)][np.minimum(h.sum(axis=0),CAP)]
def run(f):
    out=[];o=0
    for h in hs: out.append(np.flatnonzero(f(h))+o); o+=h.shape[1]
    return np.concatenate(out)
def tm(f):
    v=run(f); ts=[]
    for _ in range(R):
        s=time.perf_counter(); run(f); ts.append(time.perf_counter()-s)
    ts=np.array(ts)*1e3; return v,float(np.median(ts)),float(ts.max()-ts.min())
res={n:tm(f) for n,f in [("floored MAD, sort row 3",mad_sort),("plain binomial LUT",binom),("beta-binomial LUT + phi fit",betabin)]}
base=res["floored MAD, sort row 3"][1]
mx=max(int(h.sum(axis=0).max()) for h in hs); clamp=sum(int((h.sum(axis=0)>CAP).sum()) for h in hs)
fits=[GRID[min(fit_bin(h),len(GRID)-1)] for h in hs]; under=sum(1 for f,p in zip(fits,phis) if f<p)
print(f"numpy {np.__version__} seed {seed} tensors {len(tensors)} blocks {NBLK} P {P} CAP {CAP} grid {len(GRID)}")
print(f"LUT build {t_build*1e3:.0f} ms, table bytes {LUT.nbytes}; hist bytes {sum(h.nbytes for h in hs)}; max 8-day total {mx}, clamped {clamp}; grid phi below true phi in {under}/{len(hs)} tensors")
for n,(v,med,spr) in res.items():
    fa=len(set(v.tolist())-planted); caught=len(planted&set(v.tolist()))
    print(f"{n:30s} {med:7.2f} ms (spread {spr:5.2f})  {base/med:5.2f}x  alarms {len(v):5d}  false {fa:5d} ({fa/(NBLK-len(planted))*100:.4f}%)  caught {caught}/{len(planted)}")
```

## Results

| path | seed 202610250555 ms (spread) | seed 777 ms (spread) | vs floored MAD | false alarms (share of 508,794 quiet blocks) | planted caught |
|---|---|---|---|---|---|
| floored MAD, sort row 3 | 39.02 (6.40) | 37.15 (6.69) | 1.00x | 2,270 / 2,031 (0.446% / 0.399%) | 6/6, 6/6 |
| plain binomial lookup | 3.36 (0.75) | 3.29 (0.35) | 11.61x / 11.30x | 6,204 / 5,029 (1.219% / 0.988%) | 6/6, 6/6 |
| beta-binomial lookup + phi fit | 14.85 (7.66) | 14.11 (5.87) | 2.63x / 2.63x | 0 / 0 | 6/6, 6/6 |

One-time table build 4.4 to 4.5 s. The rounded-up grid value landed below the true phi in 8 and 11 of 210 tensors (the moment fit is noisy) without causing a false alarm. Max 8-day total 4,269 and 4,657, so 2 and 15 blocks clamped to the 4,096 row.

## Verdict

**Viable.** On overdispersed days the beta-binomial lookup is 2.63x faster than the floored MAD alarm (about 14 to 15 ms vs 37 to 39 ms for the whole 2B4T model) and takes false alarms from about 2,000 to 0, with all 6 planted blocks caught on both seeds. The plain binomial lookup stays fast but is the worst of the three here (5,000 to 6,200 false alarms), so last run's 15x path should not ship without the dispersion term. Most of the beta-binomial cost is the per-tensor variance fit, which could be stored once per day alongside the histograms.

Caveats: synthetic gamma-mixed Poisson days with independent per-block day factors; tensor-wide shared day effects or drift are not modeled. Raise the table cap to 8,192 for high-rate tensors (clamping is conservative toward fewer alarms only when the row is monotone, not checked here).

## Pair files this wave

- No quant pair push is visible for the 05:05 wave by 06:05 MT in JuniorLLM, JuniorQuant, BitNet-mlx, JuniorStock or stocksnode, and I can't see the Chief's 05:05 assignment from this lane. web3node has one unattributed cloudcover95 commit, [trit hysteresis commit @ 5a322f5](https://github.com/cloudcover95/web3node/commit/5a322f5) (03:19 MT), not touched here.
- Builds on pair A's 2026-10-24 05:05 files: [JuniorLLM scripts/daily_audit_1645_i2s_trit_flip_count_key.py @ 8a1b9fa](https://github.com/cloudcover95/JuniorLLM/blob/8a1b9fa/scripts/daily_audit_1645_i2s_trit_flip_count_key.py) (Trit Pack) and [JuniorQuant src/daily_audit_1645_f32_gamma_receipt.py @ 55ac5fb](https://github.com/cloudcover95/JuniorQuant/blob/55ac5fb/src/daily_audit_1645_f32_gamma_receipt.py) (Winsor).
- Paths honored: JuniorLLM rails/ ([ed1b5c7](https://github.com/cloudcover95/JuniorLLM/commit/ed1b5c7)) and the LLC's JuniorHome scripts/ and ports/ commits from the last 6 hours (84fc6e9, fc1a583, 7e152ad, 44867ae, 4161a2b, 2b1de78, cfed9da, 66d94c1, 98b044b, fdcfa36); this note touches only JuniorLLM findings.

Previous: [[2026-10-25_0255_daily-audit-binomial-tail-lut-vs-floored-mad]]

## Next question

Can the per-tensor dispersion fit be precomputed and stored as one fp16 or e8 scalar per tensor per day (so the alarm drops back to one sum, one gather and one compare), and does that reach at least 1.5x over the 14 ms fitted beta-binomial path with the same 0 false alarms, including when a tensor-wide shared day factor is added?

close: 4475c54e42ec9a2b (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-25_0255)
