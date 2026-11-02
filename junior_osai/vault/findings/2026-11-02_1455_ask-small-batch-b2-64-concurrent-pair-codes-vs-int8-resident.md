---
local: true
date: 2026-11-02
wave: "2026-11-02 14:05 (bench 14:55, ran 14:58 to 15:05)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` scoring a small batch of concurrent queries against the packed index)"
verdict: not-viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, ask, gemm, batching, concurrency, bench]
---

# juniorctl ask, 2 to 64 concurrent queries, index resident: no crossover, int8 trits stay faster

## Question

The 10:55 note ([[2026-11-02_1055_ask-b1-interactive-latency-pair-codes-vs-int8-resident]]) showed that for one interactive query (B = 1) int8 trits beat every pair-code path. Its next question assumed pair codes already win at B = 64; that was not right for a resident index (the 2026-11-01 02:55 note had int8 GEMM 1.19x faster than pair unpack in memory at B = 64; pair codes only won in `night` through smaller disk reads). So the real question is: when `ask` scores several users' queries together (B = 2, 4, 8, 16, 32, 64) with the index resident at 1M and 2M rows on 8 threads, is there any batch size where pair unpack + GEMM is at least 1.5x faster than int8 + GEMM, or even faster at all?

## Method

Same synthetic 384-dim index and encoders as the 10:55 run (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), built in memory. Each batch of B queries is scored together per 8,192-row chunk (one unpack or cast per chunk, shared by all B queries, then an N x 384 by 384 x B GEMM), top-10 per query per chunk, merged. OpenBLAS pinned to 1 thread, 8 worker threads. 16 timed batches per cell after one warm-up batch; the first two batches of every cell are checked query by query against an exact single-threaded int8 reference. Two seeds at 2M rows, the second run in reverse order (largest B first, pair before int8). 4M rows was skipped because MemAvailable was about 2.4 GB at start of the run (it recovered to 5.4 to 5.8 GB once the builds started). numpy 2.5.4, 8 vCPU shared box, python3 and numpy only, no network, nothing bound, no files written. A 131,072-row smoke run passed first.

```
python3 bench.py 131072 1 4 8 1,2,8 int8,pair-unpack                                   # smoke
for N in 1048576 2097152; do python3 bench.py $N 202611021455 16 8 1,2,4,8,16,32,64 int8,pair-unpack; done
python3 bench.py 2097152 202611021456 16 8 64,32,16,8,4,2,1 pair-unpack,int8               # second seed, reversed
```

Script (`bench.py`, scratch folder; imports `tern`, `enc_pair`, `V10`, `D`, `PW`, `NC`, `TOP` from the 2026-10-30 14:55 scratch script and `build`, `memavail` from the 2026-11-02 10:55 scratch script shown in that note):

```python
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("OMP_NUM_THREADS","1")
import sys, time, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,"/workspace/bench1455_1030")
from bench import tern, enc_pair, V10, D, PW, NC, TOP
import importlib.util as _u
_sp=_u.spec_from_file_location("b1055","/workspace/bench1055_1102/bench.py"); _m=_u.module_from_spec(_sp); _sp.loader.exec_module(_m)
build, memavail = _m.build, _m.memavail
C=8192; W=PW*10
def search(kind,X,P,Qb,ex):
    N=len(X); nch=-(-N//C); QT=np.ascontiguousarray(Qb.T)
    def f(c):
        a,b=c*C,min(N,(c+1)*C)
        if kind=="int8": S=X[a:b].astype(np.float32)@QT
        else: S=np.take(V10,P[a:b],axis=0).reshape(b-a,W)[:,:D].astype(np.float32)@QT
        j=np.argpartition(-S,TOP,axis=0)[:TOP]
        return np.take_along_axis(S,j,0), j+a
    parts=list(ex.map(f,range(nch))) if ex else [f(c) for c in range(nch)]
    S=np.concatenate([p[0] for p in parts]); I=np.concatenate([p[1] for p in parts])
    o=np.argsort(-S,axis=0,kind="stable")[:TOP]
    return np.take_along_axis(I,o,0).T
if __name__=="__main__":
    N=int(sys.argv[1]); seed=int(sys.argv[2]); R=int(sys.argv[3]); T=int(sys.argv[4]); Bs=[int(b) for b in sys.argv[5].split(",")]; plan=sys.argv[6].split(",")
    X,P,Q=build(N,seed); ex=ThreadPoolExecutor(T) if T>1 else None
    for B in Bs:
        batches=[Q[(i*B)%64:(i*B)%64+B] if (i*B)%64+B<=64 else Q[:B] for i in range(R)]
        refs=[search("int8",X,P,Qb,None) for Qb in batches[:2]]
        for kind in plan:
            search(kind,X,P,batches[0],ex); ts=[]; m=[]
            for i,Qb in enumerate(batches):
                t0=time.perf_counter(); I=search(kind,X,P,Qb,ex); ts.append(time.perf_counter()-t0)
                if i<2: m.append(np.mean([len(set(I[k])&set(refs[i][k]))/TOP for k in range(B)]))
            ts=np.array(ts)*1e3
            print(B, kind, np.median(ts), np.percentile(ts,10), np.percentile(ts,90), np.median(ts)/B, np.mean(m))
```

## Results

Median ms per batch over 16 batches, p10 to p90 in brackets. Top-10 matched the int8 reference exactly (100.00%) in every checked batch. "pair vs int8" is int8 time divided by pair time (above 1.0x would mean pair codes are faster).

| B | 1M int8 | 1M pair | 1M pair vs int8 | 2M int8 (seed ...455) | 2M pair | 2M pair vs int8 | 2M int8 (seed ...456, reversed) | 2M pair | 2M pair vs int8 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 22.2 (20.3-24.3) | 33.8 (33.1-35.1) | 0.66x | 43.7 (42.6-44.5) | 69.4 (68.6-71.7) | 0.63x | 42.0 (40.6-45.0) | 71.1 (67.9-74.4) | 0.59x |
| 2 | 27.9 (27.4-29.1) | 41.3 (40.4-49.2) | 0.68x | 57.2 (55.7-60.0) | 85.5 (84.1-88.9) | 0.67x | 57.8 (56.1-61.4) | 82.9 (81.7-84.9) | 0.70x |
| 4 | 29.9 (29.3-32.5) | 43.9 (42.9-44.8) | 0.68x | 60.9 (57.3-63.5) | 83.2 (82.5-85.2) | 0.73x | 58.6 (57.6-60.0) | 86.0 (83.0-96.2) | 0.68x |
| 8 | 33.6 (32.8-35.4) | 47.0 (45.7-48.4) | 0.71x | 67.7 (65.4-71.0) | 95.0 (93.0-101.3) | 0.71x | 64.8 (63.2-66.8) | 89.7 (88.5-94.8) | 0.72x |
| 16 | 42.5 (41.2-43.4) | 56.7 (55.0-58.4) | 0.75x | 89.4 (84.2-106.7) | 110.9 (107.0-115.2) | 0.81x | 82.3 (80.6-84.8) | 111.5 (108.2-118.6) | 0.74x |
| 32 | 82.3 (78.0-88.2) | 94.0 (91.9-96.5) | 0.88x | 163.4 (159.4-167.4) | 184.1 (181.6-191.4) | 0.89x | 160.7 (157.0-165.5) | 192.0 (187.1-204.1) | 0.84x |
| 64 | 176.0 (171.7-180.3) | 185.3 (179.3-189.3) | 0.95x | 344.5 (332.9-383.6) | 368.4 (354.2-379.8) | 0.94x | 331.1 (319.2-350.2) | 362.8 (355.0-373.3) | 0.91x |

Per query, batching itself is the real win for both formats: at 1M rows int8 goes from 22.2 ms (B = 1) to 2.7 ms (B = 16, 8.3x), and pair unpack from 33.8 to 3.5 ms (9.5x). Past B = 16 per-query cost stops falling (2.6 to 2.9 ms per query per 1M rows for both), because the GEMM, not the unpack or cast, now dominates. The fixed per-chunk unpack (about 13 to 14 ms per 1M rows over the int8 cast) is shared by more queries as B grows, so the gap narrows from 0.6x to about 0.93x, but it approaches parity from below and never crosses.

## Verdict

**Not viable.** With the index resident, pair unpack + GEMM is slower than int8 + GEMM at every batch size from 1 to 64, at 0.59x to 0.95x of int8 speed, on both seeds and both run orders. There is no crossover: the unpack is a fixed cost that batching only hides, and the GEMM is identical for both formats. Results are exact on both paths. For `ask`, the useful lever is batching concurrent queries (about 8x per query by B = 16 on either format); scoring should stay on int8 trits while the index fits in RAM, and the 4.92x smaller pair index matters only when it does not (the `night` disk results).

Caveats: synthetic embeddings; numpy temporary arrays and the interpreter add overhead a compiled kernel would not; shared 8 vCPU box, so differences under about 10% are noise; 4M rows not run this time because of low free memory at start.

## Pair files this wave

- No pair file for the 14:05 wave reached this run, and no new roadmap item is named; the roadmap blobs have had nothing uncovered since 2026-10-24, and the ledger will confirm the Chief's call. JuniorLLM's tip at clone time was still [263d6f3](https://github.com/cloudcover95/JuniorLLM/commit/263d6f3) (this lane's 10:55 note), so nothing new landed in JuniorLLM since 11:10 MT; this note adds one new file under `junior_osai/vault/findings/` only. The `ask` side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

The `ask` thread is closed for a resident index: int8 wins at every B. The one place pair codes could still help `ask` is when the index does not fit in RAM. On a memory-capped run (index sized past MemAvailable, read through the page cache with np.memmap from a local file under /tmp), does a B = 8 `ask` batch on pair codes beat int8 by 1.5x on latency, the way `night` did on disk?

close: 358dfda3e229efa0 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-02_1055)
