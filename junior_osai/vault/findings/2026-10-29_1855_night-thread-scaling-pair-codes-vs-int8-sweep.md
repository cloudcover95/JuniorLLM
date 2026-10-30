---
local: true
date: 2026-10-29
wave: "2026-10-29 18:05 (bench 18:55, ran 18:58 to 19:10)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`night` sweeping the packed ask index)"
verdict: not-viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, night, ask-index, threads, page-cache, bench]
---

# juniorctl night: thread scaling of the ask-index sweep, int8 trits vs uint16 pair codes

## Question

The 14:55 note ([[2026-10-29_1455_security-hit-weighted-int8-slice-vs-round-robin]]) asked how clustered the real index is, and said to switch lanes if no real hit data was reachable locally. None is: there is no `night` sweep output or packed ask index on this box, only synthetic bench files. So this run takes the fallback, with a twist the earlier `night` notes left open. All of them ran one thread ([[2026-10-27_1455_night-base3-lut-vs-dense-int8-gemm-baseline]], [[2026-10-27_1855_night-fused-pair-decode-vs-single-byte-unpack]], [[2026-10-28_1055_night-page-cache-pressure-pair-memmap-vs-int8-reread]]). When the sweep is spread over all 8 cores, it should move toward being memory-bound. Does the 4.9x smaller pair-code index (78 B/row against 384 B/row) then pay for its decode and beat int8 storage by at least 1.5x, warm or cold?

## Method

There are 2,000,000 unique rows (no tiling): 384-dim vectors around 256 topic centres, ternarized with the same 0.5 mean|x| absmean rule, and written to two files. One stores int8 trits (768,000,000 bytes). The other stores the uint16 pair codes from the 10-27 18:55 encoder (156,000,000 bytes, two 5-trit base-3 groups per code). There are 64 queries per sweep, the `night` batch.

Both paths use the same chunk-parallel sweep. A `ThreadPoolExecutor` with T workers runs over 2,048-row chunks of a memmap, and OpenBLAS is pinned to 1 thread so that the only parallelism is the worker threads. Per chunk, the int8 path casts to float32. The pair path decodes with `np.take` on the 590 KB int8 pair table, then casts. Each path then does a float32 `@` and a local top-10, and the per-chunk results are merged. T is 1, 2, 4 and 8.

Each cell has 5 repeats with alternating int8 and pair order. Warm means one untimed sweep first. Cold means `posix_fadvise(DONTNEED)` on both files before every timed sweep. Every timed sweep's top-10 indices were checked against a 1-thread int8 reference. numpy 2.5.4 (OpenBLAS 0.3.34), 8-vCPU shared box, MemAvailable about 6.5 GB, python3 and numpy only, no network, nothing bound. The index files were deleted afterwards. A 100,000-row, 2-repeat smoke run (`python3 bench.py 100000 1 2 1,4`) came first. Command, in a scratch folder (not committed):

```
python3 bench.py 2000000 202610291855 5 1,2,4,8     # N, seed, repeats, thread counts
```

Full script:

```python
# juniorctl night: thread scaling of the nightly ask-index sweep, int8 trits (384 B/row) vs uint16 pair codes (78 B/row)
# python3 + numpy only. Chunk-parallel: T Python worker threads, OpenBLAS pinned to 1 thread, each worker does
# (decode or cast) + float32 `@` + local top-10 over 2,048-row chunks; results merged. Warm (page cache) and cold (fadvise DONTNEED).
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("OMP_NUM_THREADS","1")
import sys, time, numpy as np
from concurrent.futures import ThreadPoolExecutor
D=384; C=2048; PW=39; TOP=10; NC=256
P3=3**np.arange(5)
B5=np.array([[(x//3**i)%3 for i in range(5)] for x in range(243)])
V5i=np.where(B5==1,1,np.where(B5==2,-1,0)).astype(np.int8)
V10=np.ascontiguousarray(np.concatenate([np.repeat(V5i,243,0),np.tile(V5i,(243,1))],1))   # 59,049 x 10 int8 pair table (590 KB)
def enc_pair(T):
    code=np.where(T==1,1,np.where(T==-1,2,0)).astype(np.uint16); n=len(T)
    b=(np.concatenate([code,np.zeros((n,1),np.uint16)],1).reshape(n,77,5)*P3).sum(2)
    bp=np.concatenate([b,np.zeros((n,1),np.uint16)],1); return (bp[:,0::2]*243+bp[:,1::2]).astype(np.uint16)
def tern(X):
    X/=np.linalg.norm(X,axis=1,keepdims=True); a=np.abs(X)
    return np.where(a>0.5*a.mean(1,keepdims=True),np.sign(X),0).astype(np.int8)
def build(N,seed,fp):
    rng=np.random.default_rng(seed); Cc=rng.standard_normal((NC,D)).astype(np.float32)
    with open(fp["int8"],"wb") as fi, open(fp["pair"],"wb") as fq:
        for s in range(0,N,65536):
            n=min(65536,N-s); T=tern(Cc[rng.integers(0,NC,n)]+0.9*rng.standard_normal((n,D)).astype(np.float32))
            T.tofile(fi); enc_pair(T).tofile(fq)
        for f in (fi,fq): f.flush(); os.fsync(f.fileno())
    Q=Cc[rng.integers(0,NC,64)]+0.9*rng.standard_normal((64,D)).astype(np.float32)
    return (Q/np.linalg.norm(Q,axis=1,keepdims=True)).astype(np.float32)
def chunk_int8(mm,Q,c):
    X=mm[c*C:(c+1)*C].astype(np.float32); return topk(Q@X.T,c)
def chunk_pair(mm,Q,c):
    X=np.take(V10,mm[c*C:(c+1)*C],axis=0).reshape(-1,PW*10)[:,:D].astype(np.float32); return topk(Q@X.T,c)
def topk(S,c):
    j=np.argpartition(-S,TOP,1)[:,:TOP]; return np.take_along_axis(S,j,1),j+c*C
def sweep(fp,kind,Q,T,N):
    if kind=="int8": mm=np.memmap(fp["int8"],np.int8,"r",shape=(N,D)); f=chunk_int8
    else: mm=np.memmap(fp["pair"],np.uint16,"r",shape=(N,PW)); f=chunk_pair
    nch=-(-N//C)
    with ThreadPoolExecutor(T) as ex: parts=list(ex.map(lambda c:f(mm,Q,c),range(nch)))
    S=np.concatenate([p[0] for p in parts],1); I=np.concatenate([p[1] for p in parts],1)
    o=np.argsort(-S,1,kind="stable")[:,:TOP]; del mm; return np.take_along_axis(I,o,1)
def evict(fp):
    for p in fp.values():
        fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
if __name__=="__main__":
    N=int(sys.argv[1]); seed=int(sys.argv[2]); R=int(sys.argv[3]); TS=[int(x) for x in sys.argv[4].split(",")]
    fp={"int8":"idx_int8.bin","pair":"idx_pair.bin"}
    print(f"numpy {np.__version__} N {N:,} seed {seed} repeats {R} threads {TS}; OPENBLAS_NUM_THREADS {os.environ['OPENBLAS_NUM_THREADS']}; MemAvailable {open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]} kB; nproc {os.cpu_count()}",flush=True)
    t=time.perf_counter(); Q=build(N,seed,fp); print(f"build {time.perf_counter()-t:.0f}s; bytes int8 {os.path.getsize(fp['int8']):,} pair {os.path.getsize(fp['pair']):,}",flush=True)
    ref=sweep(fp,"int8",Q,1,N); assert np.array_equal(sweep(fp,"pair",Q,1,N),ref)
    print("mode | threads | int8 median ms (spread) | pair median ms (spread) | int8/pair | top-10 match")
    for mode in ("warm","cold"):
        for T in TS:
            res={"int8":[],"pair":[]}; ok=True
            for r in range(R):
                for kind in (("int8","pair") if r%2==0 else ("pair","int8")):
                    if mode=="cold": evict(fp)
                    else: sweep(fp,kind,Q,T,N) if r==0 else None
                    t=time.perf_counter(); I=sweep(fp,kind,Q,T,N); res[kind].append(time.perf_counter()-t); ok&=np.array_equal(I,ref)
            a,b=res["int8"],res["pair"]
            print(f"{mode} | {T} | {np.median(a)*1e3:.0f} ({np.ptp(a)*1e3:.0f}) | {np.median(b)*1e3:.0f} ({np.ptp(b)*1e3:.0f}) | {np.median(a)/np.median(b):.2f}x | {'exact' if ok else 'MISMATCH'}",flush=True)
    [os.remove(p) for p in fp.values()]
```

## Results

Sweep of 2,000,000 rows by 64 queries, top-10, median ms over 5 repeats (spread = max minus min):

| mode | threads | int8 (768 MB) | pair codes (156 MB) | int8 / pair | top-10 vs reference |
|---|---|---|---|---|---|
| warm | 1 | 1312 (40) | 1463 (88) | 0.90x | exact |
| warm | 2 | 729 (38) | 789 (58) | 0.92x | exact |
| warm | 4 | 402 (10) | 446 (60) | 0.90x | exact |
| warm | 8 | 261 (25) | 273 (15) | 0.95x | exact |
| cold | 1 | 1924 (164) | 1624 (114) | 1.18x | exact |
| cold | 2 | 971 (64) | 879 (135) | 1.10x | exact |
| cold | 4 | 642 (76) | 499 (53) | **1.29x** | exact |
| cold | 8 | 480 (86) | 375 (59) | 1.28x | exact |

Scaling from 1 to 8 threads: int8 is 5.0x warm and 4.0x cold, and pair codes are 5.4x warm and 4.3x cold. In the 100k-row smoke run, the 4-thread cold ratio was 1.72x, but it did not hold at 2M rows, so that small-index number should not be relied on.

## Verdict

**Not viable.** Spreading the sweep over 8 cores narrows the warm gap: pair codes go from 0.90x to 0.95x of int8's speed, because the decode parallelizes slightly better than the cast. They still never beat int8 warm. The best cold result is 1.29x at 4 threads, which is under the 1.5x bar. On this box, re-reading 768 MB from disk is still cheaper than the extra gather work the pair decode adds to each chunk, even with all cores busy.

The `night` rule stands. Use int8 storage when it fits in RAM, and use pair codes only when int8 cannot fit at all. The reason for pair codes is capacity, never speed. One new piece of practical advice: run the sweep chunk-parallel over cores. It gives about 5x on either format with exact top-10 and no format change. That is a threading result, not a ternary one.

Caveats: synthetic embeddings, one seed, a shared and noisy machine. The virtio disk may be backed by a host cache, so cold reads may be faster than a real SD card or slow volume would be, and a slow disk is where pair codes could still cross 1.5x.

## Pair files this wave

- No pair file was visible to this run for the 18:05 wave. JuniorLLM has not moved since the 14:55 note ([9926059](https://github.com/cloudcover95/JuniorLLM/commit/9926059)), and no commit in the last 6 hours touched a path other than `junior_osai/vault/findings/`. This note adds one new file there only. The ask/night side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Find the disk speed at which pair codes cross 1.5x. Rerun the 8-thread cold sweep while capping the read rate in-process (pread each chunk with a sleep so total throughput is about 400, 200, 100 and 50 MB/s, which brackets SD cards and slow volumes). Report the throughput where int8 / pair first reaches 1.5x, then check whether any target device in the hardware lane reads that slowly.

close: b4e8cec60097a8bf (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-29_1455)
