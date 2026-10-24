---
local: true
date: 2026-10-24
wave: "2026-10-24 10:05 (bench 10:55)"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: 16:45 daily audit (where day-over-day trit flips land)"
verdict: viable
tags: [ternary, i2s, daily-audit, trit-flips, per-block, histogram, popcount, bench]
---

# Daily audit: per-block trit flip histogram on I2_S packs vs int8

## Question

Last run ([[2026-10-24_0555_daily-flip-split-sign-zero-i2s-vs-int8]]) showed the one-pass sign/zero flip split is about 3x faster on I2_S packs. The 16:45 audit also wants to know where flips land, so a hot row stands out. Can a per-block (sign, zero) flip histogram, per 128 or per 4096 trits, be computed straight from the I2_S XOR words with a reshape and a row sum of bitwise_count, and is it still at least 1.5x faster than the int8 equivalent at 17.7M trits, with identical per-block counts?

## Method

Same packing as last run: codes t+1 in {0,1,2}, four per byte, viewed as uint64 words, so a 128-trit block is 4 words and a 4096-trit block is 128 words. XOR the two days' packs, reshape to (blocks, words per block), and take row sums of popcount(x & M) for zero crossings and popcount((x >> 1) & ~x & M) for sign flips, M = 0x5555.... The int8 baseline reshapes the trits to (blocks, B) and counts t0 != t1 and t0*t1 == -1 per row. Synthetic Gaussian weights (std 0.02) with Gaussian updates at 1e-2 and 1x weight std, plus one planted hot region (trits 4096-8191 sign-inverted) so the histogram has something to find. 4.2M and 17.7M (FFN-sized) trits; median of 15 repeats after a warm-up, spread is max minus min. python3 + numpy only, no network, nothing bound.

Because np.count_nonzero with axis= is a slow path, a second check (`fair.py`) re-ran the 17.7M case against a stronger int8 baseline that sums the boolean masks as uint8 rows instead.

Command: `python3 bench.py` (inline below), numpy 2.5.4, Python 3.13.5.

```python
import numpy as np, time, sys
rng=np.random.default_rng(20261024_1055); R=15
def ternarize(w):
    g=np.float32(np.abs(w).mean(dtype=np.float64)); return np.clip(np.rint(w/(g+np.float32(1e-8))),-1,1).astype(np.int8)
def pack(t):
    c=(t+1).astype(np.uint8).reshape(-1,4)
    return (c[:,0]|(c[:,1]<<2)|(c[:,2]<<4)|(c[:,3]<<6)).astype(np.uint8)
M=np.uint64(0x5555555555555555); ONE=np.uint64(1)
def hist_packed(p0,p1,B):   # per-block (sign, zero) flip counts; B trits = B/32 uint64 words
    x=np.bitwise_xor(p0.view(np.uint64),p1.view(np.uint64)).reshape(-1,B//32)
    z=np.bitwise_count(x&M).sum(axis=1,dtype=np.int32)
    s=np.bitwise_count((x>>ONE)&~x&M).sum(axis=1,dtype=np.int32)
    return s,z
def hist_int8(t0,t1,B):
    a=t0.reshape(-1,B); b=t1.reshape(-1,B)
    tot=np.count_nonzero(a!=b,axis=1).astype(np.int32)
    s=np.count_nonzero((a*b)==-1,axis=1).astype(np.int32)
    return s,tot-s
def tm(f,*a,r=R):
    f(*a); ts=[]
    for _ in range(r):
        s=time.perf_counter(); v=f(*a); ts.append(time.perf_counter()-s)
    ts=np.array(ts)*1e3; return v,float(np.median(ts)),float(ts.max()-ts.min())
print(f"numpy {np.__version__} python {sys.version.split()[0]} R={R}")
print("| n | block trits | update/weight std | blocks | flips total | max block flips | int8 hist ms (spread) | I2_S hist ms (spread) | hist bytes (2x int32) | day bytes int8 / I2_S | I2_S vs int8 | match |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for n in (4_194_304, 17_694_720):
    w0=rng.standard_normal(n,dtype=np.float32)*np.float32(0.02)
    for us in (1e-2,1.0):
        w1=w0+rng.standard_normal(n,dtype=np.float32)*np.float32(0.02*us)
        # plant one hot block region (a "hot row") so per-block view matters
        w1[4096:8192]=-w0[4096:8192]
        t0,t1=ternarize(w0),ternarize(w1); p0,p1=pack(t0),pack(t1)
        for B in (128,4096):
            vi,mi,si=tm(hist_int8,t0,t1,B); vp,mp,sp=tm(hist_packed,p0,p1,B)
            ok=bool(np.array_equal(vi[0],vp[0]) and np.array_equal(vi[1],vp[1]))
            tot=vp[0]+vp[1]; nb=n//B
            print(f"| {n:,} | {B} | {us:.0e} | {nb:,} | {int(tot.sum()):,} | {int(tot.max()):,} | {mi:.2f} ({si:.2f}) | {mp:.2f} ({sp:.2f}) | {nb*8:,} | {n:,} / {n//4:,} | {mi/mp:.2f}x | {ok} |"); sys.stdout.flush()
```

## Results

| n | block trits | update/weight std | blocks | flips total | max block flips | int8 hist ms (spread) | I2_S hist ms (spread) | hist bytes (2x int32) | day bytes int8 / I2_S | I2_S vs int8 | match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4,194,304 | 128 | 1e-02 | 32,768 | 27,425 | 96 | 5.48 (16.81) | 1.52 (0.68) | 262,144 | 4,194,304 / 1,048,576 | 3.60x | True |
| 4,194,304 | 4096 | 1e-02 | 1,024 | 27,425 | 2,810 | 4.63 (1.42) | 0.57 (0.34) | 8,192 | 4,194,304 / 1,048,576 | 8.07x | True |
| 4,194,304 | 128 | 1e+00 | 32,768 | 1,722,581 | 96 | 5.39 (0.79) | 1.68 (0.52) | 262,144 | 4,194,304 / 1,048,576 | 3.22x | True |
| 4,194,304 | 4096 | 1e+00 | 1,024 | 1,722,581 | 2,810 | 4.14 (1.37) | 0.68 (0.31) | 8,192 | 4,194,304 / 1,048,576 | 6.11x | True |
| 17,694,720 | 128 | 1e-02 | 138,240 | 106,822 | 99 | 24.62 (17.73) | 5.80 (2.05) | 1,105,920 | 17,694,720 / 4,423,680 | 4.25x | True |
| 17,694,720 | 4096 | 1e-02 | 4,320 | 106,822 | 2,774 | 20.22 (16.00) | 2.43 (0.62) | 34,560 | 17,694,720 / 4,423,680 | 8.31x | True |
| 17,694,720 | 128 | 1e+00 | 138,240 | 7,265,720 | 99 | 21.12 (4.83) | 5.80 (1.41) | 1,105,920 | 17,694,720 / 4,423,680 | 3.64x | True |
| 17,694,720 | 4096 | 1e+00 | 4,320 | 7,265,720 | 2,774 | 18.01 (14.74) | 2.36 (1.10) | 34,560 | 17,694,720 / 4,423,680 | 7.63x | True |

Stronger int8 baseline (bool mask viewed as uint8, row sum), 17.7M trits, update 1e-2 x weight std, separate seed:

| n | block trits | update/weight std | int8 sum-hist ms (spread) | I2_S hist ms (spread) | I2_S vs int8 | match |
|---|---|---|---|---|---|---|
| 17,694,720 | 128 | 1e-02 | 14.81 (1.91) | 5.71 (1.03) | 2.59x | True |
| 17,694,720 | 4096 | 1e-02 | 10.99 (6.34) | 2.51 (0.64) | 4.38x | True |

Every per-block sign and zero count matches the int8 path exactly in all 10 rows, so error is 0. The planted hot region shows up as the max block (96-99 of 128 trits, 2,774-2,810 of 4,096) against a background of under one flip per 128-trit block at 1e-2 updates.

## Verdict

**Viable.** At 17.7M trits the per-block histogram on I2_S packs takes 2.36-2.43 ms at 4096-trit blocks and 5.80 ms at 128-trit blocks, against 18.0-24.6 ms for the plain int8 path (3.64-8.31x) and 11.0-14.8 ms for the stronger int8 row-sum baseline (2.59-4.38x). Counts are exact and the audit still keeps 4x less per day (4,423,680 vs 17,694,720 bytes). The histogram itself is small: 34,560 bytes per FFN tensor at 4096-trit blocks, 1,105,920 at 128.

Cost of locality: going from one total split (about 2.6 ms last run) to a 4096-block histogram is free, and 128-block granularity costs about 2.2x because of the short-row reductions. So 4096-trit blocks (one row of a 4096-wide matrix) is the sweet spot for the 16:45 audit; drop to 128 only to localise a row already flagged. Synthetic weights; single-thread, memory-bound timings.

## Pair files this wave

- No Trit Pack, Winsor, Side-Commit or Handshake push was seen in JuniorLLM, JuniorQuant, BitNet-mlx, JuniorHome, JuniorStock, web3node or stocksnode after 10:05 MT when this ran (about 11:00 MT). The only commit after 10:05 is JuniorHome dd11a1f (10:14 MT, ROADMAP Order #1 T13+ juniorctl skill-pin, LLC lane), which this note does not touch.
- Builds on pair A's 05:05 files: [JuniorLLM scripts/daily_audit_1645_i2s_trit_flip_count_key.py @ 8a1b9fa](https://github.com/cloudcover95/JuniorLLM/blob/8a1b9fa/scripts/daily_audit_1645_i2s_trit_flip_count_key.py) (Trit Pack) and [JuniorQuant src/daily_audit_1645_f32_gamma_receipt.py @ 55ac5fb](https://github.com/cloudcover95/JuniorQuant/blob/55ac5fb/src/daily_audit_1645_f32_gamma_receipt.py) (Winsor).
- Paths honored: Trit Pack's JuniorLLM `scripts/` file and LLC JuniorHome `scripts/` (594bd97, b686967, a44fd59, 701727b, b613952, dd0103f, dd11a1f); this note touches only JuniorLLM findings.

Previous: [[2026-10-24_0555_daily-flip-split-sign-zero-i2s-vs-int8]]

## Next question

The histogram flags hot blocks, but the audit needs a threshold that does not fire on normal drift. With per-4096-block flip counts kept daily as int32 (34,560 bytes per tensor), is a 7-day rolling median/MAD alarm on those counts cheap enough to run every day, and how does its cost and false-alarm rate compare to re-deriving the same blocks from f32 weights?

close: ca71549a6c09a30f (sha256 of previous note text || this note text, first 16 hex; prev 2026-10-24_0555)
