---
local: true
date: 2026-10-25
wave: "2026-10-25 02:05 (bench 02:55, ran 03:06)"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: 16:45 daily audit (alarm on hot trit-flip blocks)"
verdict: viable
tags: [ternary, i2s, daily-audit, trit-flips, poisson, binomial, lookup-table, alarm, whole-model, bench]
---

# Daily audit: exact count-tail lookup thresholds vs the floored median/MAD alarm (BitNet 2B4T shapes)

## Question

Last run ([[2026-10-24_1855_daily-audit-whole-model-stacked-alarm-vs-per-tensor-loop]]) found the floored 7-day median/MAD alarm false-alarms on about 0.02% of quiet blocks model-wide, mostly in low-rate tensors where the sqrt(median) floor is too thin for a count tail. Can a precomputed int32 lookup from a block's own counts to an exact tail cutoff at p = 1e-7 cut model-wide false alarms to near zero, still catch the planted hot blocks, and run at least 1.5x faster than the best floored-MAD path (per-tensor loop with the sort-row-3 median)?

## Method

Same synthetic whole model as last run: BitNet b1.58 2B4T ternary shapes, 30 layers x (q, k, v, o, gate, up, down) = 210 tensors, 508,800 blocks of 4,096 trits, 8 days of int32 per-block flip histograms (what the I2_S XOR+popcount path produces exactly). Each tensor's rate is log-uniform in 2 to 400 flips per block per day, days are Poisson, and 6 tensors (every 37th from tensor 5) get one hot block on the last day raised by 8 sqrt(rate) + 20.

Three alarm paths, all per-tensor loops (the faster layout last run):

1. Floored MAD with the sort-row-3 median (k = 5), the baseline.
2. Poisson plug-in lookup: rate = 7-day sum / 7; alarm when today > cutoff[7-day sum], where cutoff is the smallest c with P(Poisson(rate) > c) <= 1e-7.
3. Conditional binomial lookup: under a steady rate, today given the 8-day total N is Binomial(N, 1/8) whatever the rate is; alarm when today > cutoff[N] with P(Bin(N, 1/8) > cutoff) <= 1e-7. This needs no rate estimate at all.

Both tables cover totals 0 to 8,192 (32,772 bytes each, built once with exact log-pmf tails from `math.lgamma`; totals above the cap clamp to the last entry, and the largest total seen was 3,377). 21 timed repeats after one warm-up; median and max-min spread. Two seeds. numpy 2.5.4, Python 3.13.5, 8 cores, no network, nothing bound.

Commands, from a scratch dir: `python3 bench.py` and `python3 bench.py 777`.

```python
import numpy as np, time, math, sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 20261025_0255
rng=np.random.default_rng(seed); R=21; B=4096; D=8; P=1e-7
H_, KV, FF, L = 2560, 640, 6912, 30
shapes=[("q",H_,H_),("k",H_,KV),("v",H_,KV),("o",H_,H_),("gate",H_,FF),("up",H_,FF),("down",FF,H_)]
tensors=[(f"L{l}.{n}",a*b//B) for l in range(L) for n,a,b in shapes]
hs=[]; planted=set(); off=0
for i,(name,nb) in enumerate(tensors):
    lam=float(np.exp(rng.uniform(np.log(2),np.log(400))))
    h=rng.poisson(lam,size=(D,nb)).astype(np.int32)
    if i%37==5:
        j=int(rng.integers(nb)); h[-1,j]+=int(8*np.sqrt(lam)+20); planted.add(off+j)
    hs.append(h); off+=nb
NBLK=off
# ---- lookup tables (built once, timed separately) ----
CAP=1<<13
t0=time.perf_counter()
lg=np.array([math.lgamma(i+1) for i in range(CAP+1)])
def tail_cut(logpmf):
    pm=np.exp(logpmf-logpmf.max()); pm/=pm.sum()
    tail=np.cumsum(pm[::-1])[::-1]          # tail[k]=P(X>=k)
    ok=np.flatnonzero(tail<=P); return int(ok[0]-1) if ok.size else len(pm)-1  # alarm if today > cut
# (a) conditional binomial: today | 8-day total N ~ Bin(N,1/8) under a steady rate
binlut=np.empty(CAP+1,np.int32); lp,lq=math.log(1/8),math.log(7/8)
for N in range(CAP+1):
    k=np.arange(N+1); binlut[N]=tail_cut(lg[N]-lg[k]-lg[N-k]+k*lp+(N-k)*lq)
t_bin=time.perf_counter()-t0
# (b) Poisson plug-in: rate = 7-day sum / 7, indexed by 7-day sum
t0=time.perf_counter()
poilut=np.empty(CAP+1,np.int32); K=np.arange(CAP+1)
for S in range(CAP+1):
    lam=max(S/7.0,1e-3); kmax=int(lam+60*math.sqrt(lam)+80); k=K[:min(kmax,CAP)+1]
    poilut[S]=tail_cut(k*math.log(lam)-lam-lg[k])
t_poi=time.perf_counter()-t0
# ---- alarm paths (per-tensor loop, the faster layout from 18:55) ----
def mad_sort(h):
    hist,today=h[:-1],h[-1]
    s=np.sort(hist,axis=0); med=s[3].astype(np.float64)
    d=np.sort(np.abs(hist-med),axis=0); mad=d[3]
    return today>med+5.0*np.maximum(1.4826*mad,np.sqrt(np.maximum(med,1.0)))
def poisson_lut(h):
    s7=h[:-1].sum(axis=0); return h[-1]>poilut[np.minimum(s7,CAP)]
def binom_lut(h):
    n=h.sum(axis=0); return h[-1]>binlut[np.minimum(n,CAP)]
def run(f):
    out=[];o=0
    for h in hs: out.append(np.flatnonzero(f(h))+o); o+=h.shape[1]
    return np.concatenate(out)
def tm(f):
    v=run(f); ts=[]
    for _ in range(R):
        s=time.perf_counter(); run(f); ts.append(time.perf_counter()-s)
    ts=np.array(ts)*1e3; return v,float(np.median(ts)),float(ts.max()-ts.min())
res={n:tm(f) for n,f in [("floored MAD, sort row 3",mad_sort),("Poisson plug-in LUT",poisson_lut),("binomial today|8-day-total LUT",binom_lut)]}
base=res["floored MAD, sort row 3"][1]
print(f"numpy {np.__version__} seed {seed} tensors {len(tensors)} blocks {NBLK} P {P} CAP {CAP}")
print(f"LUT build: binomial {t_bin*1e3:.0f} ms, poisson {t_poi*1e3:.0f} ms, bytes {binlut.nbytes} each; hist bytes {sum(h.nbytes for h in hs)}")
print("max 8-day total seen", max(int(h.sum(axis=0).max()) for h in hs))
for n,(v,med,spr) in res.items():
    fa=len(set(v.tolist())-planted); caught=len(planted&set(v.tolist()))
    print(f"{n:34s} {med:7.2f} ms (spread {spr:5.2f})  {base/med:5.2f}x  alarms {len(v):4d}  false {fa:4d} ({fa/(NBLK-len(planted))*100:.4f}%)  caught {caught}/{len(planted)}")
```

## Results

| path | seed 202610250255 ms (spread) | seed 777 ms (spread) | vs floored MAD | alarms | false alarms (share of 508,794 quiet blocks) | planted caught |
|---|---|---|---|---|---|---|
| floored MAD, sort row 3 | 47.23 (9.81) | 46.09 (23.09) | 1.00x | 99 / 93 | 93 / 87 (0.018% / 0.017%) | 6/6, 6/6 |
| Poisson plug-in lookup | 2.94 (0.83) | 3.07 (0.46) | 16.08x / 14.99x | 7 / 7 | 1 / 1 (0.0002%) | 6/6, 6/6 |
| binomial today given 8-day total lookup | 3.17 (4.46) | 3.06 (0.55) | 14.90x / 15.05x | 6 / 6 | 0 / 0 | 6/6, 6/6 |

One-time table build: binomial 546 to 573 ms, Poisson 244 to 266 ms. Tables 32,772 bytes each next to 16,281,600 bytes of 8-day histograms (and 4.17 GB of I2_S packs or 66.7 GB of f32 for the same 8 days, per last run).

## Verdict

**Viable.** The conditional binomial lookup is about 15x faster than the best floored-MAD path (about 3 ms vs 46 to 47 ms for the whole 2B4T model) and takes false alarms from 87 to 93 down to 0, with all 6 planted blocks caught on both seeds. It replaces two sorts per tensor with one column sum, one gather and one compare. The Poisson plug-in lookup is just as fast but leaves one false alarm per seed, because the 7-day rate estimate is itself noisy; the binomial form avoids estimating the rate.

Caveat: the synthetic days are stationary Poisson. Real flip counts may drift or be overdispersed between days, which would raise binomial false alarms, so the 16:45 audit should log the observed alarm rate on real stored histograms before trusting p = 1e-7 as-is.

## Pair files this wave

- No quant pair push is visible for the 02:05 wave by 03:06 MT in JuniorLLM, JuniorQuant, BitNet-mlx, JuniorStock, web3node or stocksnode, and I can't see the Chief's 02:05 assignment from this lane.
- The only new JuniorLLM commit since the last note is [rails/trit @ ed1b5c7](https://github.com/cloudcover95/JuniorLLM/commit/ed1b5c7) (02:45 MT, offline trit rate gate, rails/ lane like LLC f536136). This note does not touch it.
- Builds on pair A's 05:05 files from 2026-10-24: [JuniorLLM scripts/daily_audit_1645_i2s_trit_flip_count_key.py @ 8a1b9fa](https://github.com/cloudcover95/JuniorLLM/blob/8a1b9fa/scripts/daily_audit_1645_i2s_trit_flip_count_key.py) (Trit Pack) and [JuniorQuant src/daily_audit_1645_f32_gamma_receipt.py @ 55ac5fb](https://github.com/cloudcover95/JuniorQuant/blob/55ac5fb/src/daily_audit_1645_f32_gamma_receipt.py) (Winsor).
- Paths honored: JuniorLLM rails/ (ed1b5c7) and the LLC's JuniorHome scripts/ and ports/ commits from the last 6 hours (fea1943, 59fb6d8, fdcfa36, 98b044b, 66d94c1, cfed9da, 2b1de78, 4161a2b); this note touches only JuniorLLM findings.

Previous: [[2026-10-24_1855_daily-audit-whole-model-stacked-alarm-vs-per-tensor-loop]]

## Next question

Real flip counts are probably overdispersed day to day. Does a beta-binomial lookup (one dispersion parameter per tensor, fitted from the stored histograms) keep false alarms near zero on overdispersed synthetic days where the plain binomial lookup breaks, at the same roughly 3 ms whole-model cost?

close: 1866edf7d5c99193 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-24_1855)
