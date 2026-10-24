---
local: true
date: 2026-10-24
wave: "2026-10-24 05:05 (bench 05:55)"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: 16:45 daily audit (day-over-day trit flips)"
verdict: viable
tags: [ternary, i2s, daily-audit, trit-flips, popcount, bench]
---

# Daily audit: split trit flips into sign flips and zero crossings on I2_S packs vs int8

## Question

Last run ([[2026-10-24_0255_daily-trit-flip-count-i2s-xor-popcnt-vs-int8]]) showed the total day-over-day flip count is exact and cheaper on I2_S packs. Can the same packs also split the flips by kind in one pass, sign flips (-1 to +1 and back) versus zero crossings (to or from 0), and stay at least 1.5x faster than an int8 path computing all three counts, with identical counts? Trit Pack built the stdlib version of this split this wave, so the bench also times that stdlib big-int path.

## Method

I2_S-style codes are t+1 in {0,1,2}, four per byte. XOR the two days' packs as uint64 words. A change to or from 0 always sets the low bit of the 2-bit field, and -1 versus +1 (codes 0 and 2) sets only the high bit. So zero = popcount(x & M), sign = popcount((x >> 1) & ~x & M), total = sign + zero, with M = 0x5555... The int8 baseline is total = count_nonzero(t0 != t1), sign = count_nonzero(t0 * t1 == -1), zero = total - sign. The stdlib path is the same masks on Python ints with int.bit_count, as in Trit Pack's file. Synthetic Gaussian weights (std 0.02) with Gaussian updates at 1e-3 to 1 times weight std, at 4.2M and 17.7M (FFN-sized) trits; median of 15 repeats (5 for the big-int path) after a warm-up, spread is max minus min. python3 + numpy only, no network.

Command: `python3 bench.py` (inline below), numpy 2.5.4, Python 3.13.5.

```python
import numpy as np, time, sys
rng=np.random.default_rng(20261024); R=15
def ternarize(w):
    g=np.float32(np.abs(w).mean(dtype=np.float64)); return np.clip(np.rint(w/(g+np.float32(1e-8))),-1,1).astype(np.int8)
def pack(t):
    c=(t+1).astype(np.uint8).reshape(-1,4)
    return (c[:,0]|(c[:,1]<<2)|(c[:,2]<<4)|(c[:,3]<<6)).astype(np.uint8)
M=np.uint64(0x5555555555555555); ONE=np.uint64(1)
def split_packed(p0,p1):
    x=np.bitwise_xor(p0.view(np.uint64),p1.view(np.uint64))
    z=int(np.bitwise_count(x&M).sum(dtype=np.int64))
    s=int(np.bitwise_count((x>>ONE)&~x&M).sum(dtype=np.int64))
    return s+z,s,z
def split_int8(t0,t1):
    d=t0!=t1; tot=int(np.count_nonzero(d))
    s=int(np.count_nonzero((t0*t1)==-1))   # -1<->+1
    return tot,s,tot-s
def split_bigint(b0,b1,m):   # Trit Pack 8a1b9fa style, stdlib int.bit_count
    x=b0^b1; z=(x&m).bit_count(); s=((x>>1)&~x&m).bit_count(); return s+z,s,z
def tm(f,*a,r=R):
    f(*a); ts=[]
    for _ in range(r):
        s=time.perf_counter(); v=f(*a); ts.append(time.perf_counter()-s)
    ts=np.array(ts)*1e3; return v,float(np.median(ts)),float(ts.max()-ts.min())
print(f"numpy {np.__version__} python {sys.version.split()[0]} R={R}")
print("| n | update/weight std | total | sign | zero | int8 3-count ms (spread) | I2_S numpy ms (spread) | I2_S stdlib bigint ms (spread) | bytes/day int8 / I2_S | numpy vs int8 | bigint vs int8 | match |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for n in (4_194_304, 17_694_720):
    w0=rng.standard_normal(n,dtype=np.float32)*np.float32(0.02)
    for us in (1e-3,1e-2,1e-1,1.0):
        w1=w0+rng.standard_normal(n,dtype=np.float32)*np.float32(0.02*us)
        t0,t1=ternarize(w0),ternarize(w1); p0,p1=pack(t0),pack(t1)
        b0=int.from_bytes(p0.tobytes(),'little'); b1=int.from_bytes(p1.tobytes(),'little')
        m=int.from_bytes(b'\x55'*(n//4),'little')
        vi,mi,si=tm(split_int8,t0,t1); vp,mp,sp=tm(split_packed,p0,p1); vb,mb,sb=tm(split_bigint,b0,b1,m,r=5)
        ok=(vi==vp==vb)
        print(f"| {n:,} | {us:.0e} | {vp[0]:,} | {vp[1]:,} | {vp[2]:,} | {mi:.2f} ({si:.2f}) | {mp:.2f} ({sp:.2f}) | {mb:.2f} ({sb:.2f}) | {n:,} / {n//4:,} | {mi/mp:.2f}x | {mi/mb:.2f}x | {ok} |"); sys.stdout.flush()
```

## Results

| n | update/weight std | total | sign | zero | int8 3-count ms (spread) | I2_S numpy ms (spread) | I2_S stdlib bigint ms (spread) | bytes/day int8 / I2_S | numpy vs int8 | bigint vs int8 | match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4,194,304 | 1e-03 | 2,567 | 0 | 2,567 | 1.47 (0.21) | 0.56 (0.11) | 1.32 (0.10) | 4,194,304 / 1,048,576 | 2.64x | 1.12x | True |
| 4,194,304 | 1e-02 | 24,762 | 0 | 24,762 | 1.47 (0.18) | 0.55 (0.14) | 1.46 (0.27) | 4,194,304 / 1,048,576 | 2.66x | 1.00x | True |
| 4,194,304 | 1e-01 | 246,399 | 0 | 246,399 | 1.45 (0.26) | 0.55 (0.11) | 1.32 (11.43) | 4,194,304 / 1,048,576 | 2.64x | 1.09x | True |
| 4,194,304 | 1e+00 | 1,722,428 | 207,923 | 1,514,505 | 1.45 (0.24) | 0.51 (0.08) | 1.56 (0.18) | 4,194,304 / 1,048,576 | 2.85x | 0.93x | True |
| 17,694,720 | 1e-03 | 10,370 | 0 | 10,370 | 7.85 (2.68) | 2.57 (0.35) | 6.04 (1.21) | 17,694,720 / 4,423,680 | 3.05x | 1.30x | True |
| 17,694,720 | 1e-02 | 103,891 | 0 | 103,891 | 7.88 (2.82) | 2.55 (0.38) | 6.87 (0.96) | 17,694,720 / 4,423,680 | 3.10x | 1.15x | True |
| 17,694,720 | 1e-01 | 1,038,571 | 0 | 1,038,571 | 6.53 (2.95) | 2.74 (0.73) | 6.12 (0.81) | 17,694,720 / 4,423,680 | 2.39x | 1.07x | True |
| 17,694,720 | 1e+00 | 7,260,751 | 872,490 | 6,388,261 | 8.50 (5.59) | 2.61 (0.72) | 7.13 (6.80) | 17,694,720 / 4,423,680 | 3.25x | 1.19x | True |

All 8 rows give identical (total, sign, zero) across the three methods, so error is 0. Bytes per day are what the audit keeps from yesterday.

## Verdict

**Viable.** The one-pass split on I2_S packs is 2.39-3.25x faster than the int8 three-count path at 17.7M trits (2.55-2.74 ms vs 6.53-8.50 ms) and 2.64-2.85x at 4.2M, with exact counts and 4x less to keep per day (4,423,680 vs 17,694,720 bytes). Getting the split costs the packed path about 1.7x its own total-only time from last run (about 2.6 ms vs 1.5 ms at 17.7M), while the int8 path pays about 3.4x (about 7.9 ms vs 2.3 ms), which is why the gap widened from 1.55x to about 3x.

Trit Pack's stdlib big-int path (JuniorLLM 8a1b9fa) is only on par with int8 numpy (0.93-1.30x), so it is fine as a stdlib receipt but a numpy uint64 path is the fast one if the audit ever runs on full tensors.

Caveat worth flagging for the audit: with realistic daily updates (up to 0.1x weight std) there were zero sign flips; every flip was a zero crossing. Sign flips only showed up at update std equal to weight std (12% of flips). So a nonzero sign count is itself a strong anomaly signal for the 16:45 audit. Synthetic weights; memory-bound single-thread timings.

## Pair files this wave

- Chief assignment (05:05): pair A, I2_S buddy file for Winsor [JuniorQuant src/daily_audit_1645_f32_gamma_receipt.py @ 55ac5fb](https://github.com/cloudcover95/JuniorQuant/blob/55ac5fb/src/daily_audit_1645_f32_gamma_receipt.py). Trit Pack pushed [JuniorLLM scripts/daily_audit_1645_i2s_trit_flip_count_key.py @ 8a1b9fa](https://github.com/cloudcover95/JuniorLLM/blob/8a1b9fa/scripts/daily_audit_1645_i2s_trit_flip_count_key.py) (05:29 MT), the stdlib split timed above.
- No Winsor, Side-Commit or Handshake push seen in JuniorQuant, JuniorLLM, BitNet-mlx or JuniorHome after 05:05 MT when this ran (about 06:00 MT).
- Paths honored: Trit Pack's JuniorLLM `scripts/` file and LLC JuniorHome `scripts/` (594bd97, aa881e1, a454fce, adb401f, 784bac9, aedab1f); this note touches only JuniorLLM findings.

Previous: [[2026-10-24_0255_daily-trit-flip-count-i2s-xor-popcnt-vs-int8]]

## Next question

The 16:45 audit wants to know where flips land, not just how many. Can a per-block flip histogram (for example per 128 or 4096 trits, so a hot row stands out) be computed from the I2_S XOR words with a reshape-and-sum of bitwise_count, and is it still at least 1.5x faster than the int8 equivalent at 17.7M, with identical per-block counts?

close: 5aa7983c4e4fdd95 (sha256 of previous note text || this note text, first 16 hex; prev 2026-10-24_0255)
