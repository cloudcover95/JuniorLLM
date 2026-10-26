---
local: true
date: 2026-10-26
wave: "2026-10-26 10:05 (bench 10:55, ran 11:00)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl ask` / `night` / `security` (scan kernel of a resident 2-bit ternary ask index)"
verdict: viable
tags: [osai, vault, findings, ternary, i2s, retrieval, juniorctl, ask, lut, gather, memory, bench]
---

# juniorctl ask: 16-bit vs 8-bit lookup-table scan over a resident 2-bit ternary index

## Question

The 2026-10-18 05:55 note ([[2026-10-18_0555_ask-index-t2-bound-rerank-vs-fp32]]) found a resident 2-bit ternary ask index is 14x to 15x smaller than fp32 and exact after a bound-driven rerank, but its 8-bit lookup-table scan was about 29 ms per ask at 100k docs, several times slower than fp32 BLAS. Its next question was never run: does a 16-bit (two-byte, 8 trits) lookup table halve the gathers enough to beat the 8-bit scan by 1.5x, or does the 12.6 MB per-query table and its build time eat the gain? I also tried the obvious third option, unpacking 4,096-row blocks back to f32 or int8 and calling BLAS. The 16:45 daily-audit line had five notes in a row, so this run rotates to Live beta step 2.

## Method

Synthetic clustered unit embeddings, 384 dims (256 topic centers plus noise), 20,000 and 100,000 docs. Each row is ternarized with an absmean threshold (|x| > 0.5 mean|x|), packed 4 trits per byte (0 to 0, +1 to 1, -1 to 2), with a per-row f32 alpha. Queries are 40 noisy copies of random docs, normalized. Per query:

1. fp32 BLAS matvec over a resident fp32 index (baseline).
2. 8-bit LUT: build a 96 x 256 f32 table from the query, then 96 column gathers over the transposed packed index.
3. 16-bit LUT: build a 48 x 65,536 f32 table (12.6 MB) from the 8-bit table by an outer add, then 48 gathers over the index viewed as uint16. The build is inside the timed path.
4. Blocked unpack to f32 (and to int8) 4,096 rows at a time, then BLAS.

Every ternary path is checked against the exact ternary scores (max abs error, top-1 agreement). The answer check reranks the top 256 alpha-scaled ternary scores with the fp32 rows and compares with the true fp32 top-1. Timings are the median over 40 queries after one warm-up, with p10 to p90 spread, one BLAS thread (`OPENBLAS_NUM_THREADS=1`), three seeds run one after another. numpy 2.5.4, Python 3.13.5, no network, nothing bound.

Commands, from a scratch dir: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 bench.py 20261026_1055`, then the same with `20261026_555` and `777`.

```python
import os, sys, time, numpy as np
seed=int(sys.argv[1]) if len(sys.argv)>1 else 20261026_1055
rng=np.random.default_rng(seed); D=384; NQ=40; CH=4096
# trit code: 0->0, +1->1, -1->2 (2 bits, 4 per byte, lane k at bits 2k)
VAL=np.array([0.,1.,-1.,0.],np.float32)
V4=np.array([[VAL[(b>>(2*k))&3] for k in range(4)] for b in range(256)],np.float32)  # 256x4
def make(N):
    # clustered unit embeddings (topics + noise), like a doc index
    C=rng.standard_normal((256,D)).astype(np.float32)
    X=C[rng.integers(0,256,N)]+0.9*rng.standard_normal((N,D)).astype(np.float32)
    X/=np.linalg.norm(X,axis=1,keepdims=True)
    a=np.abs(X); th=0.5*a.mean(axis=1,keepdims=True)
    T=np.where(a>th,np.sign(X),0).astype(np.int8)
    alpha=((X*T).sum(1)/np.maximum((T*T).sum(1),1)).astype(np.float32)
    code=np.where(T==1,1,np.where(T==-1,2,0)).astype(np.uint8).reshape(N,D//4,4)
    P=(code[:,:,0]|(code[:,:,1]<<2)|(code[:,:,2]<<4)|(code[:,:,3]<<6)).astype(np.uint8)  # N x 96
    return X,T,alpha,P
def lut8(q):  # per-query table: 96 x 256 float32 (96 KB)
    return q.reshape(-1,4)@V4.T
def scan8(PT,q):
    T=lut8(q); s=T[0][PT[0]].copy()
    for j in range(1,PT.shape[0]): s+=T[j][PT[j]]
    return s
def lut16(q):  # 48 x 65536 float32 (12.6 MB): lo byte = even position, hi byte = odd
    T=lut8(q).reshape(-1,2,256)
    return (T[:,0][:,None,:]+T[:,1][:,:,None]).reshape(-1,65536)
def scan16(P16T,q):
    T=lut16(q); s=T[0][P16T[0]].copy()
    for j in range(1,P16T.shape[0]): s+=T[j][P16T[j]]
    return s
def scanblk(P,q):  # blocked unpack to f32 (4096 rows at a time) then BLAS matvec
    out=np.empty(P.shape[0],np.float32)
    for s in range(0,P.shape[0],CH):
        U=V4[P[s:s+CH]].reshape(-1,D); out[s:s+CH]=U@q
    return out
def scanblk8(P,q):  # same, unpack to int8 then BLAS on the f32 cast of one chunk
    out=np.empty(P.shape[0],np.float32); V8=V4.astype(np.int8)
    for s in range(0,P.shape[0],CH):
        U=V8[P[s:s+CH]].reshape(-1,D); out[s:s+CH]=U.astype(np.float32)@q
    return out
def tm(f,args,Q):
    f(*args,Q[0]); ts=[]; outs=[]
    for q in Q:
        t=time.perf_counter(); o=f(*args,q); ts.append(time.perf_counter()-t); outs.append(o)
    ts=np.array(ts)*1e3; return outs,float(np.median(ts)),float(np.percentile(ts,90)-np.percentile(ts,10))
print(f"numpy {np.__version__} seed {seed} threads {os.environ.get('OPENBLAS_NUM_THREADS','?')}")
for N in (20000,100000):
    X,T,alpha,P=make(N)
    PT=np.ascontiguousarray(P.T); P16T=np.ascontiguousarray(P.view(np.uint16).T)
    Tf=T.astype(np.float32)
    Q=X[rng.integers(0,N,NQ)]+0.3*rng.standard_normal((NQ,D)).astype(np.float32)
    Q/=np.linalg.norm(Q,axis=1,keepdims=True)
    ref=[Tf@q for q in Q]  # exact ternary scores (resident f32 copy of the trits, reference only)
    lut8(Q[0]); lut16(Q[0])
    t=[time.perf_counter()]; [lut8(q) for q in Q]; t8=(time.perf_counter()-t[0])/NQ*1e3
    t=[time.perf_counter()]; [lut16(q) for q in Q]; t16=(time.perf_counter()-t[0])/NQ*1e3
    rows=[("fp32 BLAS (resident fp32 index)",lambda q:X@q,(),X.nbytes),
          ("ternary 8-bit LUT gather",scan8,(PT,),P.nbytes),
          ("ternary 16-bit LUT gather",scan16,(P16T,),P.nbytes),
          ("ternary blocked unpack f32 + BLAS",scanblk,(P,),P.nbytes),
          ("ternary blocked unpack int8 + BLAS",scanblk8,(P,),P.nbytes)]
    res={}
    for name,f,args,nb in rows:
        if args==(): outs,med,spr=tm(lambda q:X@q,(),Q)
        else: outs,med,spr=tm(f,args,Q)
        err=0.0 if name.startswith("fp32") else max(float(np.abs(o-r).max()) for o,r in zip(outs,ref))
        top=sum(int(np.argmax(o)==np.argmax(r)) for o,r in zip(outs,ref))
        # exact-answer check: rerank the top 256 alpha-scaled ternary scores with the fp32 rows (rows read from the on-disk fp32 copy in practice)
        rr=0
        for o,q in zip(outs,Q):
            c=np.argpartition(-(alpha*o),256)[:256]; rr+=int(c[np.argmax(X[c]@q)]==np.argmax(X@q))
        res[name]=(med,spr,nb,err,top,rr)
    b8=res["ternary 8-bit LUT gather"][0]; bf=res["fp32 BLAS (resident fp32 index)"][0]
    print(f"N {N} D {D} queries {NQ}  lut8 build {t8:.3f} ms  lut16 build {t16:.3f} ms (table {48*65536*4} B)")
    for n,(med,spr,nb,err,top,rr) in res.items():
        print(f"  {n:36s} {med:8.3f} ms (p10-p90 {spr:6.3f})  {b8/med:5.2f}x vs LUT8  {bf/med:5.2f}x vs fp32  index {nb:>11,d} B  max|err| vs ternary {err:.2e}  top1 same as ternary ref {top}/{NQ}  fp32 top1 after top-256 rerank {rr}/{NQ}")
```

## Results

Median ms per ask (p10 to p90 spread). Speedup is the 16-bit scan against the 8-bit scan in the same run, with the 16-bit table build included.

| seed, docs | fp32 BLAS | 8-bit LUT | 16-bit LUT | 16-bit vs 8-bit | 16-bit table build | blocked unpack f32 / int8 + BLAS |
|---|---|---|---|---|---|---|
| 1055, 20k | 0.995 (0.19) | 4.647 (0.50) | 4.026 (0.55) | 1.15x | 8.45 ms (first, cold) | 19.5 / 22.6 |
| 555, 20k | 1.012 (0.11) | 4.791 (0.62) | 4.081 (0.39) | 1.17x | 5.53 ms | 20.2 / 20.5 |
| 777, 20k | 0.967 (0.08) | 4.584 (0.89) | 3.718 (0.36) | 1.23x | 5.17 ms | 18.5 / 20.2 |
| 1055, 100k | 4.707 (1.14) | 20.519 (0.93) | 12.107 (0.61) | **1.69x** | 4.56 ms | 97.4 / 108.2 |
| 555, 100k | 5.152 (1.62) | 22.018 (1.17) | 12.457 (1.07) | **1.77x** | 4.75 ms | 98.7 / 106.8 |
| 777, 100k | 5.076 (1.62) | 21.253 (0.92) | 12.080 (1.07) | **1.76x** | 4.19 ms | 94.1 / 99.6 |

The build column comes from a separate loop over the 40 queries. Its first value includes cold-start effects; inside the scan the build is warm and costs about 4 to 5 ms.

Index bytes: fp32 30.7 MB at 20k and 153.6 MB at 100k; the packed ternary index is 1.92 MB and 9.6 MB (16x smaller), the same for both LUT widths. The 16-bit path adds a 12.6 MB table that lives only for one query. The 8-bit table is 96 KB.

Error: both LUT scans match the exact ternary scores to within 2.4e-6 (f32 summation order), and their top-1 matches the ternary reference on 240 of 240 queries. Raw ternary top-1 matches fp32 top-1 on only 5 to 11 of 40 queries, so a rerank is needed. A fixed top-256 fp32 rerank recovers the fp32 answer on 40/40 at 20k and on 39, 40 and 37 of 40 at 100k. The two LUT widths always agree. For an exact answer, keep the alpha/rho bound rerank from the 10-18 note.

## Verdict

**Viable at 100k docs.** The 16-bit table scan is 1.69x to 1.77x faster than the 8-bit scan (about 12.1 to 12.5 ms against 20.5 to 22.0 ms per ask), with identical scores and the same 9.6 MB resident index. That index is 16x smaller than fp32. At 20k docs the gain is only 1.15x to 1.23x, because the 4 to 5 ms table build is about the same size as the scan. A sensible rule is to build the 16-bit table only when the index has more than about 50k rows. Ternary still loses to one-thread fp32 BLAS on raw speed (2.4x to 2.6x slower at 100k, against 4.2x to 4.4x with the 8-bit scan), so this is a memory win that got a lot cheaper rather than a speed win over float. Blocked unpack plus BLAS is the worst option here (4.2x to 4.9x slower than the 8-bit scan), because numpy's fancy-index unpack costs more than the matvec it feeds. Synthetic embeddings only.

## Pair files this wave

- No pair report reached this run. JuniorLLM's tip at clone time was still the previous findings note [c93ed96](https://github.com/cloudcover95/JuniorLLM/commit/c93ed96), so there is no new JuniorLLM pair file for the 10:05 wave. The ask side builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) (the alpha/rho certificate) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).
- Paths honored: this note only touches JuniorLLM `junior_osai/vault/findings/`. It does not touch rails/additive/ (c5a1b99), the JuniorHome ports/ and scripts/ files pushed today (9d593eb, ee7f819, 43b11cc), or JuniorQuant 948eb95.

Previous: [[2026-10-26_0555_daily-audit-folded-rescale-rolling-sum-vs-stored-means]]

## Next question

The 8-bit table build is free (0.015 ms) and the 16-bit one costs about 4.5 ms. Would a 12-bit table (6 trits, 32 x 4,096 f32 = 512 KB, which fits in L2) keep most of the 16-bit gather saving without the build cost, so the speedup holds at 20k docs too? A second option: does a uint16 index sorted by cluster (an IVF-style probe of a few lists) cut the scan rows enough to beat fp32 BLAS outright at 100k?

close: 84aead7d0f733e9e (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-26_0555)
