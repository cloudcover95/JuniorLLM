---
local: true
date: 2026-10-26
wave: "2026-10-26 14:05 (bench 14:55, ran 15:00)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl ask` / `night` / `security` (scan kernel of a resident ternary ask index)"
verdict: viable
tags: [osai, vault, findings, ternary, base3, i2s, retrieval, juniorctl, ask, lut, gather, memory, bench]
---

# juniorctl ask: 12-bit and base-3 lookup tables vs the 8-bit and 16-bit scans

## Question

The 10:55 note ([[2026-10-26_1055_ask-index-16bit-lut-vs-8bit-lut-scan]]) found a 16-bit table scan is 1.69x to 1.77x faster than the 8-bit scan at 100k docs, but only 1.15x to 1.23x at 20k because its 12.6 MB table costs 4 to 5 ms to build per ask. Its next question: does a 12-bit table (6 trits, 64 x 4,096 f32 = 1 MiB) keep the gather saving without the build cost, so the speedup holds at 20k too? I also tried base-3 packing, which the 2-bit layout leaves on the table: 5 trits fit in one byte (3^5 = 243), and 10 trits fit in a uint16 (3^10 = 59,049). That shrinks the index from 96 to 77 bytes per row and cuts the gathers to 77 (one byte) or 39 (uint16 pairs). Quant Chief gave both pairs "no gap" at 14:05, so this run stays on the ask line that the 10:55 note opened.

## Method

Same synthetic clustered unit embeddings as the 10:55 run: 384 dims, 256 topic centers plus noise, 20,000 and 100,000 docs, absmean-threshold ternarization (|x| > 0.5 mean|x|), 40 noisy normalized queries. Every layout scores the same trits; each table is built from the query inside the timed path. Layouts:

1. fp32 BLAS matvec over the resident fp32 index (baseline for size and raw speed).
2. 2-bit pack, 8-bit table (96 x 256), 96 gathers (the 10:55 baseline).
3. 2-bit pack viewed as uint16, 16-bit table (48 x 65,536, 12.6 MB), 48 gathers.
4. 12-bit codes stored one per uint16 (128 B/row), 12-bit table (64 x 4,096, 1 MiB), 64 gathers.
5. 2-bit pack, with 12-bit codes rebuilt from 3-byte groups during the scan (96 B/row), 64 gathers.
6. Base-3 byte, 5 trits per byte (384 padded to 385, 77 B/row), 5-trit table (77 x 243), 77 gathers.
7. Base-3 uint16 pair, 10 trits per uint16 (38 columns) plus the last 5-trit byte (77 B/row), 10-trit table (38 x 59,049, 9.0 MB, built by an outer add of two 243-entry rows), 39 gathers.

Error is the max abs difference from the exact ternary scores (`T.astype(f32) @ q`) and top-1 agreement with them. Timing is the median over 40 queries after a warm-up, with the p10 to p90 spread, one BLAS thread, three seeds run one after another on an 8-core box. numpy 2.5.4, Python 3.13.5, no network, nothing bound.

Commands, from a scratch dir: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 bench.py 20261026_1455`, then the same with `20261026_555` and `777`.

```python
import os, sys, time, numpy as np
seed=int(sys.argv[1]) if len(sys.argv)>1 else 20261026_1455
rng=np.random.default_rng(seed); D=384; NQ=40
VAL=np.array([0.,1.,-1.,0.],np.float32)            # 2-bit code: 0->0, 1->+1, 2->-1
def codes_vals(bits_per,k):                          # value table for k 2-bit trits in a k*2-bit code
    c=np.arange(1<<(2*k)); return np.stack([VAL[(c>>(2*i))&3] for i in range(k)],1).astype(np.float32)
V4=codes_vals(2,4); V6=codes_vals(2,6)               # 256x4, 4096x6
P3=3**np.arange(5); B5=np.array([[(c//3**i)%3 for i in range(5)] for c in range(243)])
V5=np.where(B5==1,1.,np.where(B5==2,-1.,0.)).astype(np.float32)   # base-3 digit 1->+1, 2->-1; 243x5
def make(N):
    C=rng.standard_normal((256,D)).astype(np.float32)
    X=C[rng.integers(0,256,N)]+0.9*rng.standard_normal((N,D)).astype(np.float32)
    X/=np.linalg.norm(X,axis=1,keepdims=True)
    a=np.abs(X); th=0.5*a.mean(axis=1,keepdims=True)
    T=np.where(a>th,np.sign(X),0).astype(np.int8)
    code=np.where(T==1,1,np.where(T==-1,2,0)).astype(np.uint16)
    c4=code.reshape(N,96,4); P8=(c4[:,:,0]|(c4[:,:,1]<<2)|(c4[:,:,2]<<4)|(c4[:,:,3]<<6)).astype(np.uint8)  # N x 96 (2-bit pack)
    c6=code.reshape(N,64,6); P12=sum(c6[:,:,i]<<(2*i) for i in range(6)).astype(np.uint16)            # N x 64 uint16 (12 bits used)
    c5=np.concatenate([code,np.zeros((N,1),np.uint16)],1).reshape(N,77,5)                               # pad 384->385
    B3=(c5*P3).sum(2).astype(np.uint8)                                                                   # N x 77 base-3 bytes (<243)
    B10=(B3[:,:76].reshape(N,38,2).astype(np.uint16)@np.array([1,243],np.uint16)).astype(np.uint16)       # 38 x (10 trits) uint16 <59049
    return X,T,P8,P12,B3,B10
def scan(PT,Tab):
    s=Tab[0][PT[0]].copy()
    for j in range(1,PT.shape[0]): s+=Tab[j][PT[j]]
    return s
def lut8(q): return q.reshape(96,4)@V4.T
def s8(I,q): return scan(I,lut8(q))
def s16(I,q):
    T=lut8(q).reshape(48,2,256); return scan(I,(T[:,0][:,None,:]+T[:,1][:,:,None]).reshape(48,65536))
def s12(I,q): return scan(I,q.reshape(64,6)@V6.T)                       # 64 x 4096 f32 = 1 MiB
def s12x(I,q):                                                           # 12-bit codes made on the fly from the 2-bit pack
    B0,B1,B2=I; T=q.reshape(64,6)@V6.T; s=np.zeros(B0.shape[1],np.float32)
    for g in range(32):
        b1=B1[g]; s+=T[2*g][B0[g]|((b1&15)<<8)]; s+=T[2*g+1][(b1>>4)|(B2[g]<<4)]
    return s
def lut5(q): return np.concatenate([q,[0.]]).astype(np.float32).reshape(77,5)@V5.T   # 77 x 243
def s5(I,q): return scan(I,lut5(q))
def s10(I,q):
    BT,last=I; T=lut5(q); T10=(T[:76].reshape(38,2,243)); T10=(T10[:,0][None,:,:].transpose(1,0,2)+T10[:,1][:,:,None]).reshape(38,59049)
    s=scan(BT,T10); s+=T[76][last]; return s
def tm(f,I,Q):
    f(I,Q[0]); ts=[]; outs=[]
    for q in Q:
        t=time.perf_counter(); o=f(I,q); ts.append(time.perf_counter()-t); outs.append(o)
    ts=np.array(ts)*1e3; return outs,float(np.median(ts)),float(np.percentile(ts,90)-np.percentile(ts,10))
print(f"numpy {np.__version__} seed {seed} threads {os.environ.get('OPENBLAS_NUM_THREADS','?')}")
for N in (20000,100000):
    X,T,P8,P12,B3,B10=make(N)
    Q=X[rng.integers(0,N,NQ)]+0.3*rng.standard_normal((NQ,D)).astype(np.float32); Q/=np.linalg.norm(Q,axis=1,keepdims=True)
    Tf=T.astype(np.float32); ref=[Tf@q for q in Q]
    P8T=np.ascontiguousarray(P8.T); P16T=np.ascontiguousarray(P8.view(np.uint16).T)
    tri=P8.reshape(N,32,3); B0,B1,B2=[np.ascontiguousarray(tri[:,:,i].T).astype(np.uint16) for i in range(3)]
    rows=[("fp32 BLAS",lambda I,q:X@q,None,X.nbytes),
          ("2-bit pack, 8-bit LUT (96 gathers)",s8,P8T,P8.nbytes),
          ("2-bit pack, 16-bit LUT (48)",s16,P16T,P8.nbytes),
          ("12-bit codes in uint16, 12-bit LUT (64)",s12,np.ascontiguousarray(P12.T),P12.nbytes),
          ("2-bit pack, 12-bit codes on the fly (64)",s12x,(B0,B1,B2),P8.nbytes),
          ("base-3 byte, 5-trit LUT (77)",s5,np.ascontiguousarray(B3.T),B3.nbytes),
          ("base-3 uint16 pair, 10-trit LUT (39)",s10,(np.ascontiguousarray(B10.T),np.ascontiguousarray(B3[:,76])),B10.nbytes+N)]
    res={}
    for name,f,I,nb in rows:
        outs,med,spr=tm(f,I,Q)
        err=0.0 if f is rows[0][1] else max(float(np.abs(o-r).max()) for o,r in zip(outs,ref))
        top=sum(int(np.argmax(o)==np.argmax(r)) for o,r in zip(outs,ref))
        res[name]=(med,spr,nb,err,top)
    b8=res[rows[1][0]][0]; bf=res["fp32 BLAS"][0]
    print(f"N {N}")
    for n,(med,spr,nb,err,top) in res.items():
        print(f"  {n:42s} {med:8.3f} ms (p10-p90 {spr:6.3f}) {b8/med:5.2f}x vs LUT8 {bf/med:5.2f}x vs fp32 index {nb:>11,d} B ({nb/N:.0f} B/row) max|err| {err:.1e} top1 {top}/{NQ}")
```

## Results

Median ms per ask (p10 to p90 spread in brackets). Speedups are against the 8-bit scan in the same run.

| seed, docs | fp32 BLAS | 8-bit (96 B/row) | 16-bit (96 B/row) | 12-bit uint16 (128 B/row) | 12-bit on the fly (96 B/row) | base-3 byte (77 B/row) | base-3 uint16 pair (77 B/row) |
|---|---|---|---|---|---|---|---|
| 1455, 20k | 0.907 | 4.229 (0.53) | 3.785, 1.12x | 2.873, 1.47x | 3.318, 1.27x | 3.426, 1.23x | 2.847 (0.26), 1.49x |
| 555, 20k | 0.942 | 4.258 (0.49) | 3.807, 1.12x | 2.978, 1.43x | 3.370, 1.26x | 3.507, 1.21x | 3.696 (1.04), 1.15x |
| 777, 20k | 0.932 | 4.373 (0.60) | 3.916, 1.12x | 3.059, 1.43x | 3.334, 1.31x | 3.458, 1.26x | 3.011 (0.26), 1.45x |
| 1455, 100k | 4.912 | 20.882 (1.35) | 12.261, 1.70x | 13.738, 1.52x | 15.577, 1.34x | 16.762, 1.25x | **9.808 (1.28), 2.13x** |
| 555, 100k | 4.829 | 21.703 (9.22) | 12.801, 1.70x | 14.330, 1.51x | 16.191, 1.34x | 17.809, 1.22x | **10.025 (0.86), 2.16x** |
| 777, 100k | 4.764 | 20.749 (3.13) | 11.532, 1.80x | 14.026, 1.48x | 16.361, 1.27x | 17.230, 1.20x | **9.837 (1.83), 2.11x** |

Index bytes at 100k: fp32 153.6 MB; 2-bit pack 9.6 MB (16x under fp32); 12-bit uint16 12.8 MB (12x); base-3 7.7 MB (19.9x under fp32, 20% under the 2-bit pack). At 20k the same ratios hold (30.7 MB, 1.92 MB, 2.56 MB, 1.54 MB).

Error: every ternary layout matches the exact ternary scores to within 2.1e-6 (f32 summation order) and agrees with the ternary top-1 on 40 of 40 queries in all six runs (1,440 checks across the six layouts). Raw ternary top-1 still matches fp32 top-1 on only 8 to 13 of 40, so the rerank from the 10-18 and 10:55 notes is still needed for the final answer; this run only changes the scan kernel.

## Verdict

**Viable at 100k docs.** Base-3 packing, 10 trits per uint16 with a 10-trit table, is 2.11x to 2.16x faster than the 8-bit scan (9.8 to 10.0 ms against 20.7 to 21.7 ms per ask) and 1.17x to 1.28x faster than the 16-bit scan from the 10:55 note. It does that with a 20% smaller index (77 B/row, 7.7 MB at 100k, 19.9x under fp32) and identical scores. The 12-bit table does not answer the 20k question: stored one code per uint16 it reaches 1.43x to 1.47x at 20k, just under the bar, and costs 33% more index memory; rebuilt from the 2-bit pack during the scan it falls to 1.26x to 1.31x. At 20k no layout clears 1.5x reliably (the base-3 pair ranged 1.15x to 1.49x because its 9 MB table build dominates). Ternary is still about 2x slower than one-thread fp32 BLAS on raw speed at 100k (against 4.2x to 4.5x for the 8-bit scan), so this remains a memory win that is now much cheaper, not a speed win over float. Synthetic embeddings only.

A practical rule from both notes: store the ask index in base-3 (77 B/row). Below about 50k rows, scan it one byte at a time with the 243-entry table (1.2x over today's 8-bit scan, no build cost); above that, pair the bytes and use the 10-trit table.

## Pair files this wave

- No pair report reached this run, and Quant Chief's 14:05 assignment was "no gap" for both pairs, so there is no new pair file for the 14:05 wave. JuniorLLM's tip at clone time was still the previous findings note [051d53c](https://github.com/cloudcover95/JuniorLLM/commit/051d53c). The ask side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).
- Paths honored: this note only touches JuniorLLM `junior_osai/vault/findings/`. It does not touch JuniorHome 0cb280e (scripts/, pushed 14:11 MT), rails/additive/, or JuniorQuant 948eb95.

Previous: [[2026-10-26_1055_ask-index-16bit-lut-vs-8bit-lut-scan]]

## Next question

The base-3 pair table costs most of its time in the 38 x 59,049 outer-add build at 20k. Can the build be shared across a batch of queries (for example a `juniorctl night` sweep that asks 16 questions at once, gathering 16-wide f32 rows from a 38 x 59,049 x 16 table), so the per-ask cost at 20k drops below 2.8 ms and clears 1.5x over the 8-bit scan? A second option: does a rerank over the top candidates of the base-3 scan still hit 40/40 fp32 top-1 when the index is 7.7 MB instead of 9.6 MB (it should, since the trits are the same, but the bound rerank has not been rerun on base-3 codes)?

close: e8e368c699851c6e (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-26_1055)
