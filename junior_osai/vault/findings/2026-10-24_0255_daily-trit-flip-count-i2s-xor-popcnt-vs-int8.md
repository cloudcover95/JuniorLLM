---
local: true
date: 2026-10-24
wave: "2026-10-24 02:05"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations (current) ON: 16:45 daily audit (day-over-day trit flip count on stored I2_S packs)"
chief_assignment: "Chief 02:05 (quant-chief/last_tips.json 08:06Z): pair A Trit Pack buddy file for Winsor JuniorQuant 55ac5fb (16:45 daily audit); pair B no eligible item, no gap"
verdict: viable
tags: [ternary, osai, findings, automations, daily-audit-1645, i2s, trit-pack, popcount, xor, drift, int8, fp32, numpy, viable]
---

# Day-over-day trit flip counts on I2_S packs are exact, 4x smaller to keep than int8 and 1.55-2.0x faster to compare: viable

## Question

The last three notes all re-benched the 16:45 daily audit's gamma scalars, and the Chief flagged that as recursion. This run asks a different question on the I2_S side of the same item, which is Trit Pack's half of the pair A assignment (buddy file for Winsor JuniorQuant `55ac5fb`): if the daily audit keeps yesterday's ternary weights only as an I2_S pack (2 bits per trit), can it count how many trits flipped since yesterday straight from the two packs, with XOR plus popcount, instead of keeping int8 trits or the f32 weights and re-ternarizing? The bar is at least 1.5x less memory or time than the int8 baseline, with flip counts identical to it.

## Method

- Trits from the BitNet absmean rule: gamma = mean|W| (f64 accumulate, cast to f32), t = clip(rint(W / (gamma + 1e-8)), -1, 1).
- Pack: code = t + 1 in {0, 1, 2}, four codes per byte (I2_S-style 2-bit fields). Packs are viewed as uint64 words.
- Candidate: x = p0 XOR p1; a 2-bit field differs iff either bit is set, so flips = popcount((x | x >> 1) & 0x5555...5555) using `np.bitwise_count`.
- Baselines: int8 trits kept from yesterday, `count_nonzero(t0 != t1)`; and f32 weights kept from yesterday, re-ternarize both days and compare.
- Data: W ~ N(0, 0.02) f32 at 4,194,304 and 17,694,720 (the 2560x6912 BitNet 2B4T FFN shape); day-two W = W + N(0, s), with s / 0.02 = 1e-3, 1e-2, 1e-1, giving about 0.06%, 0.6% and 5.9% flips. Seed 255.
- 31 timed repeats after one warm-up, median and max-min spread in ms; `OMP_NUM_THREADS=1`; numpy 2.5.4, python3 only, x86_64 8 cores, no network, nothing bound.

## Command

```
R=31 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 flipbench.py
```

```python
# 16:45 daily audit: day-over-day trit flip count on I2_S packs vs int8 trits vs f32 re-ternarize
import os, time, sys, numpy as np
assert os.environ.get("OMP_NUM_THREADS")=="1"
R=int(os.environ.get("R","9")); rng=np.random.default_rng(255)
def ternarize(w):
    g=np.float32(np.abs(w).mean(dtype=np.float64)); return np.clip(np.rint(w/(g+np.float32(1e-8))),-1,1).astype(np.int8)
def pack(t):  # I2_S-style: code=t+1 in {0,1,2}, 4 codes/byte, n%32==0
    c=(t+1).astype(np.uint8).reshape(-1,4)
    return (c[:,0]|(c[:,1]<<2)|(c[:,2]<<4)|(c[:,3]<<6)).astype(np.uint8)
M=np.uint64(0x5555555555555555)
def flips_packed(p0,p1):
    x=np.bitwise_xor(p0.view(np.uint64),p1.view(np.uint64))
    return int(np.bitwise_count((x|(x>>np.uint64(1)))&M).sum(dtype=np.int64))
def flips_int8(t0,t1): return int(np.count_nonzero(t0!=t1))
def flips_f32(w0,w1): return int(np.count_nonzero(ternarize(w0)!=ternarize(w1)))
def tm(f,*a):
    f(*a); ts=[]
    for _ in range(R):
        s=time.perf_counter(); v=f(*a); ts.append(time.perf_counter()-s)
    ts=np.array(ts)*1e3; return v,float(np.median(ts)),float(ts.max()-ts.min())
print(f"numpy {np.__version__} R={R}")
print("| n | update std / weight std | flips | flip % | f32 re-tern ms (spread) | int8 != ms (spread) | I2_S xor+popcnt ms (spread) | bytes/day f32 / int8 / I2_S | speed vs int8 | speed vs f32 | match |")
for n in (4_194_304, 17_694_720):
    w0=(rng.standard_normal(n,dtype=np.float32)*np.float32(0.02))
    for us in (1e-4,1e-3,1e-2):
        w1=w0+rng.standard_normal(n,dtype=np.float32)*np.float32(0.02*us*10)
        t0,t1=ternarize(w0),ternarize(w1); p0,p1=pack(t0),pack(t1)
        vf,mf,sf=tm(flips_f32,w0,w1); vi,mi,si=tm(flips_int8,t0,t1); vp,mp,sp=tm(flips_packed,p0,p1)
        ok=(vf==vi==vp)
        print(f"| {n:,} | {us*10:.0e} | {vp:,} | {100*vp/n:.3f} | {mf:.2f} ({sf:.2f}) | {mi:.2f} ({si:.2f}) | {mp:.2f} ({sp:.2f}) | {4*n:,} / {n:,} / {n//4:,} | {mi/mp:.2f}x | {mf/mp:.0f}x | {ok} |")
        sys.stdout.flush()
```

## Results

| n | update std / weight std | flips | flip % | f32 re-tern ms (spread) | int8 != ms (spread) | I2_S xor+popcnt ms (spread) | bytes/day f32 / int8 / I2_S | speed vs int8 | speed vs f32 | match |
|---|---|---|---|---|---|---|---|---|---|---|
| 4,194,304 | 1e-03 | 2,495 | 0.059 | 17.19 (2.21) | 0.59 (0.47) | 0.33 (0.10) | 16,777,216 / 4,194,304 / 1,048,576 | 1.76x | 52x | True |
| 4,194,304 | 1e-02 | 24,668 | 0.588 | 17.63 (3.09) | 0.60 (0.39) | 0.33 (0.17) | 16,777,216 / 4,194,304 / 1,048,576 | 1.84x | 54x | True |
| 4,194,304 | 1e-01 | 245,476 | 5.853 | 13.57 (0.72) | 0.60 (0.39) | 0.30 (0.10) | 16,777,216 / 4,194,304 / 1,048,576 | 2.03x | 46x | True |
| 17,694,720 | 1e-03 | 10,409 | 0.059 | 137.60 (109.29) | 2.29 (6.26) | 1.46 (5.93) | 70,778,880 / 17,694,720 / 4,423,680 | 1.57x | 94x | True |
| 17,694,720 | 1e-02 | 104,351 | 0.590 | 137.17 (46.01) | 2.29 (2.64) | 1.48 (0.41) | 70,778,880 / 17,694,720 / 4,423,680 | 1.55x | 93x | True |
| 17,694,720 | 1e-01 | 1,035,342 | 5.851 | 126.68 (113.86) | 2.27 (2.11) | 1.47 (0.42) | 70,778,880 / 17,694,720 / 4,423,680 | 1.55x | 86x | True |

Bytes per day are what the audit has to keep from yesterday to answer the question. All 6 rows give identical flip counts across the three methods (error 0).

## Verdict

**Viable.** Keeping yesterday's weights as an I2_S pack is 4x smaller than int8 trits (4,423,680 vs 17,694,720 bytes for the FFN tensor) and 16x smaller than f32, and the XOR plus popcount count is 1.55-1.57x faster than the int8 compare at 17.7M and 1.76-2.03x faster at 4.2M, with exact counts. Against an audit that keeps f32 and re-ternarizes, it is 46-94x faster. Flip rate doesn't change the cost, since every word is read either way.

Caveats: the speed edge at 17.7M only just clears 1.5x and both int8 and I2_S times are memory-bound single-pass numbers; the real gain is the 4x storage. The count says how many trits moved, not which way; separating sign flips (-1 to +1) from zero crossings needs one more mask on the high bit of each field, not measured here. Synthetic Gaussian weights and updates.

## Pair files this wave

- Chief assignment (02:05): pair A, Trit Pack buddy file for Winsor [JuniorQuant src/daily_audit_1645_f32_gamma_receipt.py @ 55ac5fb](https://github.com/cloudcover95/JuniorQuant/blob/55ac5fb/src/daily_audit_1645_f32_gamma_receipt.py); no Trit Pack or Winsor push seen in BitNet-mlx, JuniorQuant or JuniorStock since 00:00 MT when this ran (03:10 MT).
- Pair B: Chief said no eligible item; Handshake pushed [JuniorHome docs/slices/2026-10-24-ports-handshake-index.md @ 29a4054](https://github.com/cloudcover95/JuniorHome/blob/29a4054/docs/slices/2026-10-24-ports-handshake-index.md).
- LLC skip paths honored: JuniorHome `scripts/advance_beta_to_os_owner_tz.py` (784bac9) and `scripts/` (adb401f, aedab1f); this note touches only JuniorLLM findings.

Previous: [[2026-10-23_1855_daily-gamma-hist-bracket-vs-simd-sort]]

## Next question

Can the same packs split the flips by kind in one pass: sign flips (-1 to +1 and back) versus zero crossings, using the high bit of each 2-bit field (codes 0 and 2, meaning -1 and +1, differ only in the high bit, while a change to or from code 1, meaning 0, always changes the low bit)? Is that still at least 1.5x faster than the int8 path computing both counts, with identical counts?

close: 01eb281954fe7e8f (sha256 of previous note text || this note text, first 16 hex; prev 2026-10-23_1855)
