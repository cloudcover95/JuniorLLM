---
local: true
date: 2026-10-22
wave: "2026-10-22 14:05"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations (current) ON: 16:45 daily audit (writes docs/AUTOMATION_AUDIT.md)"
chief_assignment: "16:45 daily audit assigned to pair B at 14:14 MT (Quant Chief); no pair A file seen on box this wave"
verdict: viable
tags: [ternary, osai, findings, automations, daily-audit, backfill, i2s, swar, base3, numpy, viable]
---

# 16:45 daily-audit backfill: scanning i2s 2-bit packed day receipts is 3.1-3.3x faster and 3.33x smaller than int8 rows, bit-exact; JSON text is ~45x slower than int8

## Question

Side-Commit's 16:45 audit ledger (stocksnode `4eb7e33`) folds 10 field trits per day (time, tip, audit_row, overnight, slot19, weekly, paused_absent, paused_note, no_actions, tz) into an i2s 2-bit pack (0->0, 1->1, -1->2, MSB first), and its `backfill_check` looks for failed fields and +1 -> -1 regressions across days. Should the backfill scan run directly on the packed bytes, or unpack to int8 / keep JSON rows? Does base-3 (5 trits/byte equivalent, uint16 per day) beat 2-bit?

## Method

Synthetic day receipts (p(+1)=0.85, p(0)=0.10, p(-1)=0.05, seed 1455) at 3,650 days (10 years), 365,000 and 3,650,000 days (throughput). Layouts: int8 [days,10]; i2s 2-bit packed to 3 bytes/day (12 trit slots, 2 zero pad), checked with SWAR masks on a uint32 word (fail = hi & ~lo, plus = lo & ~hi, regression = plus[d] & fail[d+1]); base-3 uint16/day decoded via a 59,049-row LUT; JSON text rows parsed with `json.loads`. Two outputs: per-field counts (fail and regression per field) and per-day flags (which days have any fail / any regression, which is what `backfill_check` reports). 15 repeats (JSON 5 at 365k), median/min/max wall time. Every layout is checked for exact equality against the int8 reference. python3 3.13.5, numpy 2.5.4, 8 cores, no network, nothing bound.

Command: `python3 /workspace/bench1455/bench.py` (script inline below).

## Results

```
days|layout|bytes|median_ms|min_ms|max_ms|exact|speed_vs_int8|bytes_vs_int8
3650|int8|36500|0.112|0.111|0.133|True|1.00x|1.00x
3650|i2s-2bit SWAR|10950|0.185|0.181|0.263|True|0.61x|3.33x
3650|i2s-2bit day-flags|10950|0.038|0.036|0.075|True|2.93x|3.33x
3650|base3-u16 LUT|7300|0.157|0.142|0.207|True|0.71x|5.00x
3650|json-text|158758|5.423|5.248|6.461|True|0.02x|0.23x
365000|int8|3650000|11.862|11.502|13.202|True|1.00x|1.00x
365000|i2s-2bit SWAR|1095000|19.912|19.453|27.111|True|0.60x|3.33x
365000|i2s-2bit day-flags|1095000|4.407|4.283|5.726|True|2.69x|3.33x
365000|base3-u16 LUT|730000|14.907|14.583|16.051|True|0.80x|5.00x
365000|json-text|15877948|525.613|518.280|541.864|True|0.02x|0.23x
3650000|int8|36500000|139.449|133.235|155.157|True|1.00x|1.00x
3650000|i2s-2bit SWAR|10950000|368.013|351.353|769.497|True|0.38x|3.33x
3650000|i2s-2bit day-flags|10950000|53.858|48.360|73.565|True|2.59x|3.33x
3650000|base3-u16 LUT|7300000|173.492|165.938|181.342|True|0.80x|5.00x
fair day-flags baseline
3650|int8 day-flags|36500|0.129|0.127|0.172|i2s 0.039ms|3.32x|exact True
365000|int8 day-flags|3650000|13.695|13.491|13.924|i2s 4.331ms|3.16x|exact True
3650000|int8 day-flags|36500000|154.333|148.057|161.827|i2s 49.484ms|3.12x|exact True
```

| days | layout | bytes | median ms | vs int8 speed | vs int8 bytes | exact |
|---|---|---|---|---|---|---|
| 3,650 | int8 day-flags | 36,500 | 0.129 | 1.00x | 1.00x | yes |
| 3,650 | i2s 2-bit day-flags (SWAR) | 10,950 | 0.039 | 3.32x | 3.33x | yes |
| 365,000 | int8 day-flags | 3,650,000 | 13.695 | 1.00x | 1.00x | yes |
| 365,000 | i2s 2-bit day-flags (SWAR) | 1,095,000 | 4.331 | 3.16x | 3.33x | yes |
| 3,650,000 | int8 day-flags | 36,500,000 | 154.333 | 1.00x | 1.00x | yes |
| 3,650,000 | i2s 2-bit day-flags (SWAR) | 10,950,000 | 49.484 | 3.12x | 3.33x | yes |
| 365,000 | i2s per-field counts (SWAR + shift-extract) | 1,095,000 | 19.912 | 0.60x | 3.33x | yes |
| 365,000 | base-3 uint16 + LUT | 730,000 | 14.907 | 0.80x | 5.00x | yes |
| 365,000 | JSON text rows | 15,877,948 | 525.613 | 0.02x | 0.23x | yes |

## Verdict

**viable** for the day-flag scan: running `backfill_check`'s failed-day and regression-day detection directly on the i2s packed bytes is 3.12-3.32x faster (medians) and 3.33x smaller than int8 rows, with zero error (exact match at all three sizes). Not worth it: per-field counts on the packed form (0.38-0.61x, the per-field shift-extract costs more than it saves), and base-3 uint16 (5.00x smaller but 0.71-0.80x the int8 speed because of the LUT gather). JSON rows are about 45x slower than int8 and 4.3x larger, so the audit text should be parsed once into the pack, not rescanned. At the real scale (10 years = 3,650 days) every layout except JSON finishes under 0.2 ms, so the gain matters for batch replays across many ledgers, not for one daily run.

## Links

- Previous note: [[2026-10-22_1055_loopback-handoff-i2s-vs-int8-fp16-f32-8gib-cap]]
- Pair B, 14:05 wave: Side-Commit cloudcover95/stocksnode `4eb7e33` daily_audit_1645_receipt_ledger.py (16:45 audit receipt ledger, i2s pack + rate lock 0.05); Handshake & Ports cloudcover95/JuniorHome `062c856` ports/daily_audit_slot_port_map.py (9 pass, 2 advisory, 0 fail)
- Pair A: no 14:05 file seen on the box.

## Next question

Does a uint64 SWAR word holding 3 days (30 trits + pad) let the regression check (day d vs d+1) stay inside one word and beat the 3-byte/day form again, or does the cross-word carry eat the gain?

## Inline script

```python
# 2026-10-22 14:55 bench: 16:45 daily-audit backfill_check on 10 field trits/day.
# int8 rows vs i2s 2-bit packed rows (SWAR regression/fail detection on packed bytes)
# vs base-3 uint16 rows vs JSON text. python3 + numpy only, no network.
import json, time, statistics as st, numpy as np
F=10; REPS=15
def mk(days, seed=1455):
    r=np.random.default_rng(seed)
    t=r.choice(np.array([1,0,-1],np.int8),size=(days,F),p=[.85,.1,.05])
    return t
def ref(t):  # failed per field, regressions (+1 -> -1 day to day) per field
    return (t==-1).sum(0), ((t[:-1]==1)&(t[1:]==-1)).sum(0)
def pack_i2s(t):  # 0->0,1->1,-1->2, 2 bits, MSB first, pad to 12 trits = 3 bytes/day
    c=np.where(t==-1,2,t).astype(np.uint32)
    c=np.concatenate([c,np.zeros((len(t),2),np.uint32)],1)
    w=np.zeros(len(t),np.uint32)
    for i in range(12): w|=c[:,i]<<np.uint32(22-2*i)
    b=np.stack([(w>>16)&255,(w>>8)&255,w&255],1).astype(np.uint8)
    return b
M_LO=np.uint32(sum(1<<(22-2*i) for i in range(F)))  # low bit of each field
def i2s_check(b):
    w=(b[:,0].astype(np.uint32)<<16)|(b[:,1].astype(np.uint32)<<8)|b[:,2]
    lo=w&M_LO; hi=(w>>np.uint32(1))&M_LO
    fail=hi&~lo                     # code 2 (-1)
    plus=lo&~hi                     # code 1 (+1)
    reg=plus[:-1]&fail[1:]
    sh=np.array([22-2*i for i in range(F)],np.uint32)
    fc=((fail[:,None]>>sh)&1).sum(0); rc=((reg[:,None]>>sh)&1).sum(0)
    return fc,rc
def i2s_check_pop(b):  # totals only via popcount (what backfill really flags: any fail/regression per day)
    w=(b[:,0].astype(np.uint32)<<16)|(b[:,1].astype(np.uint32)<<8)|b[:,2]
    lo=w&M_LO; hi=(w>>np.uint32(1))&M_LO
    fail=hi&~lo; reg=(lo&~hi)[:-1]&fail[1:]
    return np.flatnonzero(fail), np.flatnonzero(reg)
def b3(t):
    p=(3**np.arange(F)).astype(np.uint16); return ((t+1).astype(np.uint16)*p).sum(1).astype(np.uint16)
LUT3=np.array([[(v//3**i)%3-1 for i in range(F)] for v in range(3**F)],np.int8)
def b3_check(u): return ref(LUT3[u])
def json_rows(t): return [json.dumps({"fields":r.tolist()}) for r in t]
def json_check(rows):
    fc=[0]*F; rc=[0]*F; prev=None
    for s in rows:
        f=json.loads(s)["fields"]
        for i,v in enumerate(f):
            if v==-1:
                fc[i]+=1
                if prev and prev[i]==1: rc[i]+=1
        prev=f
    return np.array(fc),np.array(rc)
def tm(fn,*a,reps=None):
    xs=[]
    for _ in range(reps or REPS):
        t0=time.perf_counter(); out=fn(*a); xs.append(time.perf_counter()-t0)
    return st.median(xs),min(xs),max(xs),out
print("days|layout|bytes|median_ms|min_ms|max_ms|exact|speed_vs_int8|bytes_vs_int8")
for days in (3650, 365_000, 3_650_000):
    t=mk(days); R=ref(t)
    p=pack_i2s(t); u=b3(t)
    assert np.array_equal(LUT3[u],t)
    rows=json_rows(t) if days<=365_000 else None
    jb=sum(len(s)+1 for s in rows) if rows else None
    m8=tm(ref,t)
    res=[("int8",t.nbytes,m8,True)]
    mi=tm(i2s_check,p); res.append(("i2s-2bit SWAR",p.nbytes,mi,all(np.array_equal(a,b) for a,b in zip(mi[3],R))))
    mp=tm(i2s_check_pop,p)
    ok=np.array_equal(mp[3][0],np.flatnonzero((t==-1).any(1))) and np.array_equal(mp[3][1],np.flatnonzero(((t[:-1]==1)&(t[1:]==-1)).any(1)))
    res.append(("i2s-2bit day-flags",p.nbytes,mp,ok))
    mb=tm(b3_check,u); res.append(("base3-u16 LUT",u.nbytes,mb,all(np.array_equal(a,b) for a,b in zip(mb[3],R))))
    if rows:
        mj=tm(json_check,rows,reps=5 if days>3650 else 15)
        res.append(("json-text",jb,mj,all(np.array_equal(a,b) for a,b in zip(mj[3],R))))
    for n,by,m,ok in res:
        print(f"{days}|{n}|{by}|{m[0]*1e3:.3f}|{m[1]*1e3:.3f}|{m[2]*1e3:.3f}|{ok}|{m8[0]/m[0]:.2f}x|{t.nbytes/by:.2f}x")
print("fair day-flags baseline")
def int8_flags(t): return np.flatnonzero((t==-1).any(1)), np.flatnonzero(((t[:-1]==1)&(t[1:]==-1)).any(1))
for days in (3650,365_000,3_650_000):
    t=mk(days); p=pack_i2s(t)
    a=tm(int8_flags,t); b=tm(i2s_check_pop,p)
    ok=all(np.array_equal(x,y) for x,y in zip(a[3],b[3]))
    print(f"{days}|int8 day-flags|{t.nbytes}|{a[0]*1e3:.3f}|{a[1]*1e3:.3f}|{a[2]*1e3:.3f}|i2s {b[0]*1e3:.3f}ms|{a[0]/b[0]:.2f}x|exact {ok}")
```

close: c7a13ca5e22d0386 (sha256(previous note text || this note text above this line)[:16])
