---
local: true
date: 2026-10-24
wave: "2026-10-24 18:05 (bench 18:55, ran 19:00)"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: 16:45 daily audit (alarm on hot trit-flip blocks)"
verdict: not-viable
tags: [ternary, i2s, daily-audit, trit-flips, rolling-median, mad, alarm, whole-model, bench]
---

# Daily audit: one stacked whole-model alarm pass vs a per-tensor loop (BitNet 2B4T shapes)

## Question

Last run ([[2026-10-24_1455_daily-audit-rolling-mad-alarm-stored-hist-vs-f32-rederive]]) showed the floored 7-day median/MAD alarm on stored int32 flip histograms costs about 0.5 ms per FFN tensor. A whole model has hundreds of ternary tensors with very different flip rates. Does concatenating every tensor's (8, blocks) histogram into one stack and running a single alarm pass beat a per-tensor loop by at least 1.5x, and does the whole-model audit stay well inside the 16:45 slot?

## Method

BitNet b1.58 2B4T ternary shapes: 30 layers x (q 2560x2560, k 2560x640, v 2560x640, o 2560x2560, gate 2560x6912, up 2560x6912, down 6912x2560) = 210 tensors, 2,084,044,800 trits, 508,800 blocks of 4,096 trits (embeddings are not ternary and are left out). Histograms are synthesized directly, since the last three notes already showed I2_S packs give exact counts: each tensor gets a flip rate drawn log-uniformly from 2 to 400 flips per block per day, and its eight daily rows are Poisson at that rate. Six tensors (every 37th, starting at tensor 5) get one hot block on the last day, raised by 8 sqrt(rate) + 20.

Four paths, all using the same floored alarm (k = 5, scale = max(1.4826 MAD, sqrt(max(median, 1)))):

1. Per-tensor loop with `np.median` (baseline, what a straight port of the 14:55 script does).
2. One stacked (8, 508,800) pass with `np.median`.
3. Per-tensor loop where the 7-row median is `np.sort(axis=0)[3]` (exact for 7 rows, same float math).
4. One stacked pass with the sort-row-3 median.

21 timed repeats after one warm-up, median and max-min spread reported. Alarm lists are mapped to global block ids and compared with the baseline. numpy 2.5.4, Python 3.13.5, 8 cores, no network, nothing bound.

Command, run twice from a scratch dir: `python3 bench.py` (inline script below).

```python
import numpy as np, time, sys
rng=np.random.default_rng(20261024_1855); R=21; B=4096; D=8
H_, KV, FF, L = 2560, 640, 6912, 30
shapes=[("q",H_,H_),("k",H_,KV),("v",H_,KV),("o",H_,H_),("gate",H_,FF),("up",H_,FF),("down",FF,H_)]
tensors=[(f"L{l}.{n}",a*b//B) for l in range(L) for n,a,b in shapes]
nparams=sum(nb*B for _,nb in tensors)
def floor_alarm(hist,today,k=5.0):
    med=np.median(hist,axis=0); mad=np.median(np.abs(hist-med),axis=0)
    thr=med+k*np.maximum(1.4826*mad,np.sqrt(np.maximum(med,1.0))); return today>thr
def sort_alarm(hist,today,k=5.0):
    s=np.sort(hist,axis=0); med=s[3].astype(np.float64)
    d=np.sort(np.abs(hist-med),axis=0); mad=d[3]
    thr=med+k*np.maximum(1.4826*mad,np.sqrt(np.maximum(med,1.0))); return today>thr
def build():
    hs=[]; planted=[]
    for i,(name,nb) in enumerate(tensors):
        lam=float(np.exp(rng.uniform(np.log(2),np.log(400))))
        h=rng.poisson(lam,size=(D,nb)).astype(np.int32)
        if i%37==5: j=int(rng.integers(nb)); h[-1,j]+=int(8*np.sqrt(lam)+20); planted.append((i,j))
        hs.append(h)
    return hs,planted
hs,planted=build()
offs=np.cumsum([0]+[h.shape[1] for h in hs]); S=np.concatenate(hs,axis=1)
def loop_median(): return [np.flatnonzero(floor_alarm(h[:-1],h[-1])) for h in hs]
def stack_median(): return np.flatnonzero(floor_alarm(S[:-1],S[-1]))
def loop_sort(): return [np.flatnonzero(sort_alarm(h[:-1],h[-1])) for h in hs]
def stack_sort(): return np.flatnonzero(sort_alarm(S[:-1],S[-1]))
def tm(f):
    f(); ts=[]
    for _ in range(R):
        s=time.perf_counter(); v=f(); ts.append(time.perf_counter()-s)
    ts=np.array(ts)*1e3; return v,float(np.median(ts)),float(ts.max()-ts.min())
def to_global(lst): return np.concatenate([a+offs[i] for i,a in enumerate(lst)])
# timing, comparison and table printing as in the Results table below
```

## Results

Run 1 / run 2 (median ms, spread ms in brackets):

| path | run 1 ms | run 2 ms | vs per-tensor np.median | alarms | planted caught | same alarms as baseline |
|---|---|---|---|---|---|---|
| per-tensor loop, np.median | 74.89 (10.20) | 75.16 (5.58) | 1.00x | 120 | 6/6 | True |
| one stacked pass, np.median | 72.03 (11.19) | 68.23 (13.83) | 1.04x / 1.10x | 120 | 6/6 | True |
| per-tensor loop, sort row 3 | 37.94 (17.18) | 35.35 (6.75) | 1.97x / 2.13x | 120 | 6/6 | True |
| one stacked pass, sort row 3 | 46.55 (12.46) | 42.35 (8.39) | 1.61x / 1.77x | 120 | 6/6 | True |

Bytes for 8 days of the whole model: histograms 16,281,600, I2_S packs 4,168,089,600, f32 66,689,433,600 (histograms 256x smaller than packs, 4,096x smaller than f32).

## Verdict

**Not viable** for the question asked. One stacked pass is only 1.04-1.10x faster than looping over the 210 tensors, well under the 1.5x bar, because each tensor already has 400-4,320 blocks and numpy's per-call overhead is small next to the median work. With the faster median it is actually slower than the loop (1.61-1.77x vs 1.97-2.13x), likely because the 16 MB stack falls out of cache while per-tensor slices stay in it.

Side finding worth keeping (not a ternary gain, just an implementation choice): taking the 7-row median as `np.sort(axis=0)[3]` gives identical alarms and is about 2x faster than `np.median`. The whole-model audit costs about 35-75 ms on this box either way, so time is not a constraint for the 16:45 slot.

What does need work is the threshold. 114 of the 120 alarms are false on a model-wide scale (about 0.022% of 508,800 quiet blocks, against 0 in 43,200 on the single FFN tensor last run), and they cluster in low-rate tensors, where the sqrt(median) floor is too thin for the Poisson tail. All six planted blocks were caught on every path.

## Pair files this wave

- I can't see the Chief's 18:05 assignment from this lane. The only commit in JuniorLLM, JuniorQuant, BitNet-mlx, JuniorHome, JuniorStock, web3node or stocksnode after 18:05 MT by about 19:00 MT is [JuniorHome scripts/advance_i2sd_userspace_port.py @ 77d6d79](https://github.com/cloudcover95/JuniorHome/blob/77d6d79/scripts/advance_i2sd_userspace_port.py) (18:14 MT, author "nico juniorcloudllc", ROADMAP #3 i2sd userspace port confirmation, LLC style). This note does not touch it.
- Builds on pair A's 05:05 files: [JuniorLLM scripts/daily_audit_1645_i2s_trit_flip_count_key.py @ 8a1b9fa](https://github.com/cloudcover95/JuniorLLM/blob/8a1b9fa/scripts/daily_audit_1645_i2s_trit_flip_count_key.py) (Trit Pack) and [JuniorQuant src/daily_audit_1645_f32_gamma_receipt.py @ 55ac5fb](https://github.com/cloudcover95/JuniorQuant/blob/55ac5fb/src/daily_audit_1645_f32_gamma_receipt.py) (Winsor).
- Paths honored: Trit Pack's JuniorLLM `scripts/` file and the LLC's JuniorHome `scripts/` files from the last 6 hours (165021b, 890bf6e, 0c37e90, 1e91741, a921035, 9f4b88d, 60384b0, 77d6d79); this note touches only JuniorLLM findings.

Previous: [[2026-10-24_1455_daily-audit-rolling-mad-alarm-stored-hist-vs-f32-rederive]]

## Next question

Low-rate tensors false-alarm under the sqrt(median) floor. Does an exact Poisson upper-tail threshold per tensor rate (a small precomputed int32 lookup from median to cutoff at p = 1e-7) cut model-wide false alarms to near zero while still catching the planted blocks, and is the lookup at least as fast as the floored MAD?

close: 65f5df066f14885c (sha256 of previous note text || this note text, first 16 hex; prev 2026-10-24_1455)
