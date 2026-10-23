---
local: true
date: 2026-10-22
wave: "2026-10-22 18:05"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations (current) ON: 16:45 daily audit (writes docs/AUTOMATION_AUDIT.md)"
chief_assignment: "16:45 daily audit, remaining engines only, assigned to pair A at 18:10 MT (Quant Chief)"
verdict: viable
tags: [ternary, osai, findings, automations, daily-audit, backfill, i2s, swar, uint64, numpy, viable]
---

# 16:45 daily-audit day flags on a 3-day uint64 I2_S word are 2.4-2.6x faster than the 3-byte/day form and 7.4-8.3x faster than int8 at batch scale, exact; only 1.125x smaller

## Question

The 14:55 note asked: does packing 3 audit days (30 field trits + 2 zero pad slots) into one uint64 SWAR word, with the day d vs d+1 regression check done as an in-word 20-bit shift plus a cross-word 40-bit carry, beat the 3-byte/day uint32 form again, or does the carry eat the gain? Trit Pack built that layout this wave (JuniorLLM `ee086b7`), so this run measures it.

## Method

Synthetic day receipts with 10 field trits per day (p(+1)=0.85, p(0)=0.10, p(-1)=0.05, seed 1855) at 3,650 days (10 years), 365,000 and 3,650,000 days. Three layouts compute the same output as `backfill_check`'s day flags (days with any failed field, days d where some field goes +1 on d to -1 on d+1):

- int8 rows `[days, 10]` (reference);
- i2s 3 bytes/day, rebuilt to a uint32 per day, SWAR masks (14:55 layout);
- i2s 3 days per uint64 word, MSB-first slots (0->0, 1->1, -1->2), fail = hi & ~lo, plus = lo & ~hi, in-word regression = plus & (fail << 20) on days 0-1, cross-word regression = plus[k] day 2 & (fail[k+1] >> 40).

21 repeats per case, median/min/max wall time, every output checked for exact equality with int8, plus two edge asserts (a regression straddling a word boundary, and a partial last word). The whole script was run twice; both runs are shown. python3 3.13.5, numpy 2.5.4, 8 cores, no network, nothing bound, no files read.

Command: `python3 /workspace/bench1855/bench.py` (script inline below).

## Results

Run 1:

| days | layout | bytes | median ms | min ms | max ms | vs int8 | vs 3B/day | bytes vs 3B/day | exact |
|---|---|---|---|---|---|---|---|---|---|
| 3,650 | int8 rows | 36,500 | 0.143 | 0.127 | 0.168 | 1.00x | 0.26x | 0.30x | yes |
| 3,650 | i2s 3B/day u32 | 10,950 | 0.037 | 0.036 | 0.147 | 3.86x | 1.00x | 1.00x | yes |
| 3,650 | i2s 3-day u64 word | 9,736 | 0.031 | 0.030 | 0.102 | 4.66x | 1.21x | 1.125x | yes |
| 365,000 | int8 rows | 3,650,000 | 13.949 | 13.614 | 20.200 | 1.00x | 0.36x | 0.30x | yes |
| 365,000 | i2s 3B/day u32 | 1,095,000 | 4.958 | 4.200 | 13.282 | 2.81x | 1.00x | 1.00x | yes |
| 365,000 | i2s 3-day u64 word | 973,336 | 1.969 | 1.874 | 2.422 | 7.08x | 2.52x | 1.125x | yes |
| 3,650,000 | int8 rows | 36,500,000 | 154.851 | 146.390 | 174.547 | 1.00x | 0.32x | 0.30x | yes |
| 3,650,000 | i2s 3B/day u32 | 10,950,000 | 49.467 | 46.011 | 70.572 | 3.13x | 1.00x | 1.00x | yes |
| 3,650,000 | i2s 3-day u64 word | 9,733,336 | 19.420 | 18.970 | 25.533 | 7.97x | 2.55x | 1.125x | yes |

Run 2 (packed layouts only):

| days | layout | median ms | min ms | max ms | vs int8 | vs 3B/day | exact |
|---|---|---|---|---|---|---|---|
| 3,650 | i2s 3B/day u32 | 0.047 | 0.039 | 0.117 | 3.34x | 1.00x | yes |
| 3,650 | i2s 3-day u64 word | 0.034 | 0.032 | 0.095 | 4.65x | 1.39x | yes |
| 365,000 | i2s 3B/day u32 | 4.428 | 4.344 | 5.170 | 3.17x | 1.00x | yes |
| 365,000 | i2s 3-day u64 word | 1.698 | 1.644 | 2.125 | 8.27x | 2.61x | yes |
| 3,650,000 | i2s 3B/day u32 | 49.263 | 44.957 | 72.826 | 3.05x | 1.00x | yes |
| 3,650,000 | i2s 3-day u64 word | 20.256 | 19.749 | 26.222 | 7.42x | 2.43x | yes |

Edge asserts (cross-word carry, partial last word): pass.

## Verdict

**viable** on speed at batch scale: the 3-day uint64 word runs the day-flag scan 2.43-2.61x faster than the 3-byte/day form and 7.1-8.3x faster than int8 rows at 365k and 3.65M days, bit-exact, and the cross-word carry does not eat the gain (it is one shifted AND over n/3 words). Not a memory win over the previous packed form: 9,736 vs 10,950 bytes for 10 years is only 1.125x smaller (still 3.75x smaller than int8). At the real 3,650-day scale the speed gain over the 3-byte form is 1.21-1.39x, under the 1.5x bar, and both finish under 0.05 ms, so the gain is for batch replays across many ledgers, not one daily run. Most of the 3-byte form's cost is rebuilding the uint32 from three bytes; the uint64 word skips that and does a third as many element operations.

## Links

- Previous note: [[2026-10-22_1455_audit-backfill-i2s-day-flags-vs-int8-json]]
- Pair A, 18:05 wave: Trit Pack cloudcover95/JuniorLLM `ee086b7` scripts/daily_audit_1645_word3_i2s_key.py (3-day I2_S word pack, JTR1 frame, selftest key b1e7071185213aaa); Winsor & Quant Error cloudcover95/JuniorStock `b15f1f0` quant/daily_audit_1645_quant_error_section.py (per-engine absmean/junior/p95 quant-error rows vs analytic bound)
- Pair B: no 18:05 file seen on the box at 18:55.

## Next question

Winsor's quant-error section (`b15f1f0`) computes relative MSE and its bound per engine in float64; does doing it on the I2_S packed trits plus one f32 gamma (r = q * g decoded via a 4-entry LUT) cut the audit's error-section time or memory by 1.5x at equal error to 1e-6?

## Inline script

```python
# 2026-10-22 18:55 bench: 16:45 daily-audit day flags on a 3-day uint64 I2_S word
# (30 trits + 2 pad, in-word 20-bit shift + cross-word carry) vs the 3-byte/day
# uint32 form (14:55 note) vs int8 rows. python3 + numpy only, no network.
import time, statistics as st, numpy as np
F=10; DPW=3; REPS=21
def mk(days, seed=1855):
    r=np.random.default_rng(seed)
    return r.choice(np.array([1,0,-1],np.int8),size=(days,F),p=[.85,.1,.05])
def codes(t): return np.where(t==-1,2,t).astype(np.uint64)
# --- 3 bytes/day (14:55 layout) ---
def pack3b(t):
    c=codes(t).astype(np.uint32); w=np.zeros(len(t),np.uint32)
    for i in range(F): w|=c[:,i]<<np.uint32(22-2*i)
    return np.stack([(w>>16)&255,(w>>8)&255,w&255],1).astype(np.uint8)
M32=np.uint32(sum(1<<(22-2*i) for i in range(F)))
def flags3b(b):
    w=(b[:,0].astype(np.uint32)<<16)|(b[:,1].astype(np.uint32)<<8)|b[:,2]
    lo=w&M32; hi=(w>>np.uint32(1))&M32
    fail=hi&~lo; reg=(lo&~hi)[:-1]&fail[1:]
    return np.flatnonzero(fail), np.flatnonzero(reg)
# --- 3 days per uint64 word (Trit Pack ee086b7 layout, MSB-first slots) ---
def bit(s): return 62-2*s
def pack64(t):
    n=len(t); nw=-(-n//DPW); c=np.zeros((nw*DPW,F),np.uint64); c[:n]=codes(t)
    c=c.reshape(nw,DPW*F); w=np.zeros(nw,np.uint64)
    for s in range(DPW*F): w|=c[:,s]<<np.uint64(bit(s))
    return w, n
DM=[np.uint64(sum(1<<bit(j*F+i) for i in range(F))) for j in range(DPW)]
MLO=np.uint64(sum(1<<bit(s) for s in range(DPW*F)))
S20=np.uint64(20); S40=np.uint64(40); ONE=np.uint64(1)
def flags64(wn):
    w,n=wn
    lo=w&MLO; hi=(w>>ONE)&MLO
    fail=hi&~lo; plus=lo&~hi
    # in-word: day j+1 sits 20 bits lower; shift fail up by 20 to align with day j
    rin=plus&(fail<<S20)&(DM[0]|DM[1])
    # cross-word: day 2 of word k vs day 0 of word k+1 (40-bit shift down)
    rx=plus[:-1]&DM[2]&(fail[1:]>>S40)
    F3=np.stack([(fail&DM[j])!=0 for j in range(DPW)],1).reshape(-1)[:n]
    R=np.zeros((len(w),DPW),bool)
    R[:,0]=(rin&DM[0])!=0; R[:,1]=(rin&DM[1])!=0; R[:-1,2]=rx!=0
    R=R.reshape(-1)[:n-1]
    return np.flatnonzero(F3), np.flatnonzero(R)
def flags_i8(t): return np.flatnonzero((t==-1).any(1)), np.flatnonzero(((t[:-1]==1)&(t[1:]==-1)).any(1))
def tm(fn,a):
    xs=[]
    for _ in range(REPS):
        t0=time.perf_counter(); out=fn(a); xs.append(time.perf_counter()-t0)
    return st.median(xs),min(xs),max(xs),out
print("days|layout|bytes|median_ms|min_ms|max_ms|exact|speed_vs_int8|speed_vs_3B|bytes_vs_3B")
for days in (3650,365_000,3_650_000):
    t=mk(days); R=flags_i8(t); b=pack3b(t); w=pack64(t)
    a8=tm(flags_i8,t); a3=tm(flags3b,b); a6=tm(flags64,w)
    for name,by,m in (("int8 rows",t.nbytes,a8),("i2s 3B/day u32",b.nbytes,a3),("i2s 3day/u64 word",w[0].nbytes,a6)):
        ok=all(np.array_equal(x,y) for x,y in zip(m[3],R))
        print(f"{days}|{name}|{by}|{m[0]*1e3:.3f}|{m[1]*1e3:.3f}|{m[2]*1e3:.3f}|{ok}|{a8[0]/m[0]:.2f}x|{a3[0]/m[0]:.2f}x|{b.nbytes/by:.3f}x")
# edge: a regression straddling a word boundary must be caught
t=np.ones((6,F),np.int8); t[3,4]=-1  # day2->day3 is word0 day2 -> word1 day0
assert np.array_equal(flags64(pack64(t))[1],np.array([2])), "cross-word carry"
t=np.ones((7,F),np.int8); t[6,9]=-1  # partial last word
assert np.array_equal(flags64(pack64(t))[1],np.array([5]))
print("edge cases ok")
```

close: 6e4609cee1f4903c (sha256(previous note text || this note text above this line)[:16])
