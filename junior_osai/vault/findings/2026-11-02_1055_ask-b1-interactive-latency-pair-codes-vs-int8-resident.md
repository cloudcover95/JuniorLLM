---
local: true
date: 2026-11-02
wave: "2026-11-02 10:05 (bench 10:55, ran 11:00 to 11:20)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` scoring one interactive query against the packed index)"
verdict: not-viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, ask, gemv, lut, latency, bench]
---

# juniorctl ask, one interactive query (B = 1), index resident: int8 trits beat pair codes on latency

## Question

The 05:55 note ([[2026-11-02_0555_night-24m-rows-int8-out-of-cache-pair-resident-vs-int8]]) closed the `night` thread: on a fast disk the pair-code sweep never clears 1.5x. This run rotates to the `ask` side of Live beta step 2. With one query at a time (B = 1), scoring is a matrix-vector product, so there is no batch to amortize the pair unpack over. With the whole index resident in RAM at 1M to 4M rows, is a pair-code path at least 1.5x faster than int8 trits on per-query latency? Two pair paths are tried: the usual unpack (np.take on the 59,049 x 10 pair table, then GEMV) and a per-query score table that only fits B = 1 (precompute the dot product of every pair code with each 10-wide query slice, a 39 x 59,049 float32 table of 9.2 MB, then each row's score is 39 lookups and a sum, with no unpack at all).

## Method

Synthetic 384-dim index built in memory with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row). 8,192-row chunks, OpenBLAS pinned to 1 thread, run with 1 worker thread and with 8, top-10 per chunk merged. 32 queries per cell, each timed alone (one warm-up query first), every result checked against an exact single-threaded int8 reference. Two seeds; the second seed runs the cells in reverse order. numpy 2.5.4, 8 vCPU shared box, 16 GB RAM, MemAvailable 4.5 to 5.8 GB after each build. python3 and numpy only, no network, nothing bound, no files written. Passed a 131,072-row smoke run first.

```
python3 bench.py 131072 1 4 1,8 int8,pair-unpack,pair-lut                 # smoke
for N in 1048576 2097152 4194304; do python3 bench.py $N 202611021055 32 1,8 int8,pair-unpack,pair-lut; done
python3 bench.py 4194304 202611021056 32 8,1 pair-lut,pair-unpack,int8      # second seed, reversed order
```

Script (`bench.py`, scratch folder, imports `tern`, `enc_pair`, `V10`, `D`, `PW`, `NC`, `TOP` from the 2026-10-30 14:55 scratch script):

```python
# juniorctl ask, B = 1 interactive query, index resident in RAM: int8 trits (384 B/row) vs uint16 pair codes (78 B/row).
# Paths per 8,192-row chunk: int8 = cast + X @ q (GEMV); pair-unpack = np.take on 59,049 x 10 pair table + cast + GEMV;
# pair-lut = per-query 39 x 59,049 float32 score table (Lt[p,c] = V10[c] . q[10p:10p+10]) then gather + sum over 39 codes.
# OpenBLAS pinned to 1 thread; T worker threads; top-10 merged; every query checked against an exact int8 reference.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("OMP_NUM_THREADS","1")
import sys, time, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,"/workspace/bench1455_1030")
from bench import tern, enc_pair, V10, D, PW, NC, TOP
C=8192; W=PW*10; OFF=(np.arange(PW)*59049).astype(np.int64)
def memavail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1])//1024
def build(N,seed):
    rng=np.random.default_rng(seed); Cc=rng.standard_normal((NC,D)).astype(np.float32)
    X=np.empty((N,D),np.int8); P=np.empty((N,PW),np.uint16)
    for s in range(0,N,65536):
        n=min(65536,N-s); T=tern(Cc[rng.integers(0,NC,n)]+0.9*rng.standard_normal((n,D)).astype(np.float32))
        X[s:s+n]=T; P[s:s+n]=enc_pair(T)
    Q=Cc[rng.integers(0,NC,64)]+0.9*rng.standard_normal((64,D)).astype(np.float32)
    return X,P,(Q/np.linalg.norm(Q,axis=1,keepdims=True)).astype(np.float32)
def top(s,c): j=np.argpartition(-s,TOP)[:TOP]; return s[j],j+c*C
def search(kind,X,P,q,ex):
    N=len(X); nch=-(-N//C)
    if kind=="pair-lut":
        qp=np.zeros(W,np.float32); qp[:D]=q
        Lt=np.ascontiguousarray((V10.astype(np.float32)@qp.reshape(PW,10).T).T).ravel()   # 39*59,049 floats, 9.2 MB
    def f(c):
        a,b=c*C,min(N,(c+1)*C)
        if kind=="int8": s=X[a:b].astype(np.float32)@q
        elif kind=="pair-unpack": s=np.take(V10,P[a:b],axis=0).reshape(b-a,W)[:,:D].astype(np.float32)@q
        else: s=np.take(Lt,P[a:b].astype(np.int64)+OFF).sum(1)
        return top(s,c)
    parts=list(ex.map(f,range(nch))) if ex else [f(c) for c in range(nch)]
    S=np.concatenate([p[0] for p in parts]); I=np.concatenate([p[1] for p in parts])
    return I[np.argsort(-S,kind="stable")[:TOP]]
if __name__=="__main__":
    N=int(sys.argv[1]); seed=int(sys.argv[2]); NQ=int(sys.argv[3]); threads=[int(t) for t in sys.argv[4].split(",")]; plan=sys.argv[5].split(",")
    t0=time.perf_counter(); X,P,Q=build(N,seed); tb=time.perf_counter()-t0
    print(f"numpy {np.__version__} N {N:,} seed {seed} queries {NQ} chunk {C}; bytes int8 {X.nbytes:,} pair {P.nbytes:,} ({X.nbytes/P.nbytes:.2f}x); build {tb:.0f} s; MemAvailable {memavail()} MB",flush=True)
    Qs=Q[:NQ]
    # exact int8 reference, single-threaded chunked
    ref=[search("int8",X,P,q,None) for q in Qs]
    print("threads | path | median ms per query (p10-p90) | ms per 1M rows | top-10 vs int8")
    for T in threads:
        ex=ThreadPoolExecutor(T) if T>1 else None
        for kind in plan:
            search(kind,X,P,Qs[0],ex)  # warm-up
            ts=[]; m=[]
            for q,r in zip(Qs,ref):
                t0=time.perf_counter(); I=search(kind,X,P,q,ex); ts.append(time.perf_counter()-t0); m.append(len(set(I)&set(r))/TOP)
            ts=np.array(ts)*1e3; med=np.median(ts)
            print(f"{T} | {kind} | {med:.1f} ({np.percentile(ts,10):.1f}-{np.percentile(ts,90):.1f}) | {med*1048576/N:.1f} | {np.mean(m)*100:.2f}%",flush=True)
        if ex: ex.shutdown()
    print("done")
```

## Results

Median ms per query over 32 queries, p10 to p90 in brackets. Top-10 matched the int8 reference exactly (100.00%) in every cell. Index bytes are 384 B/row int8 vs 78 B/row pair (4.92x smaller, already recorded in earlier notes).

| rows | threads | int8 ms | pair-unpack ms | pair-lut ms | best pair vs int8 |
|---|---|---|---|---|---|
| 1,048,576 | 1 | 148.7 (146.0-159.1) | 244.2 (240.5-254.0) | 176.0 (172.8-193.5) | 0.84x (lut) |
| 1,048,576 | 8 | 21.6 (20.6-24.8) | 34.7 (34.1-35.3) | 36.6 (34.2-38.7) | 0.62x (unpack) |
| 2,097,152 | 1 | 311.8 (299.1-347.7) | 494.5 (484.5-515.9) | 371.6 (357.5-377.0) | 0.84x (lut) |
| 2,097,152 | 8 | 47.2 (43.2-51.0) | 69.1 (67.0-73.3) | 59.1 (56.1-63.2) | 0.80x (lut) |
| 4,194,304 (seed ...055) | 1 | 622.8 (597.2-652.1) | 963.0 (936.5-992.8) | 709.4 (667.9-747.3) | 0.88x (lut) |
| 4,194,304 (seed ...055) | 8 | 82.6 (79.7-85.4) | 134.7 (129.6-141.3) | 105.5 (102.1-113.4) | 0.78x (lut) |
| 4,194,304 (seed ...056, reversed) | 8 | 84.5 (82.4-98.4) | 138.4 (135.0-144.5) | 114.1 (109.2-117.1) | 0.74x (lut) |
| 4,194,304 (seed ...056, reversed) | 1 | 569.4 (563.4-581.4) | 1,002.1 (944.8-1,034.1) | 720.5 (688.1-772.2) | 0.79x (lut) |

Per million rows: int8 costs 142 to 156 ms single-threaded and 21 to 24 ms on 8 threads; the pair unpack costs 240 to 251 and 34 to 35; the score table costs 176 to 186 and 26 to 37. The table removes the unpack and beats it by about 1.3x, but 39 random gathers per row (into a 9.2 MB table that does not fit in L2) still cost more than casting 384 int8 bytes and running a dense GEMV, which streams at roughly 2.6 GB/s per core here.

## Verdict

**Not viable.** At B = 1 with the index resident, int8 trits are the fastest path at every size and thread count: the best pair-code path (the per-query score table) runs at 0.74x to 0.88x of int8 speed, and the plain unpack at 0.60x to 0.65x. Results are exact (100% top-10) on every path, so the only pair-code gain left on the `ask` side is the 4.92x smaller index, which matters only when int8 would not fit in RAM; for a resident interactive `ask`, keep int8 trits for scoring.

Caveats: synthetic embeddings; numpy gathers carry interpreter and temporary-array overhead that a compiled kernel would not, so this measures the stack as it stands, not a native ceiling; shared 8 vCPU box, so differences under about 10% are noise.

## Pair files this wave

- No pair file for the 10:05 wave reached this run, and no new roadmap item is named; the roadmap blobs have had nothing uncovered since 2026-10-24, and the ledger will confirm the Chief's call. JuniorLLM's tip at clone time was still [34fdc4b](https://github.com/cloudcover95/JuniorLLM/commit/34fdc4b) (this lane's 05:55 note), so nothing new landed in JuniorLLM since 06:24 MT; this note adds one new file under `junior_osai/vault/findings/` only. The ask side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Interactive `ask` usually waits on more than one user at once. With 2, 4 and 8 concurrent queries (small B, 1M to 4M rows resident, 8 threads), where does the pair unpack start to beat int8, given that at B = 64 it already did and at B = 1 it loses?

close: ebedb8f76a643fc3 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-02_0555)
