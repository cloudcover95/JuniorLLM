---
local: true
date: 2026-10-30
wave: "2026-10-30 05:05 (bench 05:55, ran 06:06 to 06:27)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`night` sweeping the packed ask index)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, night, ask-index, sd-card, read-latency, chunk-size, bench]
---

# juniorctl night: per-request read latency and chunk size on slow storage, int8 trits vs uint16 pair codes

## Question

The 02:55 note ([[2026-10-30_0255_night-throttled-disk-crossover-pair-codes-vs-int8]]) found pair codes 4.8x to 4.9x faster than int8 for a cold nightly sweep at 100 and 50 MB/s, but its device model charged only for bytes. Real SD cards and USB sticks also pay a fixed cost per read request. Once that per-request cost is added, does the pair-code advantage stay above 1.5x, and which chunk size should `night` use on an SD card?

## Method

Same index and sweep as the 02:55 run: 2,000,000 unique rows, 384-dim, ternarized with the 0.5 mean|x| absmean rule, stored as int8 trits (768,000,000 bytes) and as uint16 pair codes (156,000,000 bytes). 64 queries, top-10, 8 worker threads, OpenBLAS pinned to 1 thread, page cache evicted before every timed sweep.

The one change is the device model. It still serves one request at a time, but each chunk read now costs a fixed latency L plus its bytes at the capped rate R. The grid is R of 100 and 50 MB/s, L of 0.2, 1 and 5 ms per request (which brackets SD card random-read latency), and chunk sizes of 512, 2,048 and 8,192 rows. A 512-row chunk is 196,608 bytes for int8 and 39,936 bytes for pair codes, so the sweep makes 3,907 requests on either format. A 2,048-row chunk means 977 requests and an 8,192-row chunk means 245. Each cell has 3 repeats with alternating int8 and pair order. Every timed sweep's top-10 indices were checked against a 1-thread uncapped int8 reference.

numpy 2.5.4, 8-vCPU shared box, MemAvailable about 2.5 GB, python3 and numpy only, no network, nothing bound. Index files were deleted afterwards. A 100,000-row smoke run (`python3 bench.py 100000 1 1 100 0.2,5 512,8192`) came first. Command, in a scratch folder (not committed):

```
python3 bench.py 2000000 202610300555 3 100,50 0.2,1,5 512,2048,8192     # N, seed, repeats, read caps MB/s, per-request latency ms, chunk rows
```

Full script:

```python
# juniorctl night: per-request latency x chunk size on slow storage, int8 trits (384 B/row) vs uint16 pair codes (78 B/row)
# python3 + numpy only. 8 worker threads, OpenBLAS pinned to 1 thread. Each chunk is read with os.pread through a
# shared device model that serves requests one after another at R MB/s (a reservation clock under a lock; the
# reader sleeps until its slot ends, then preads). Page cache is evicted before every timed sweep. R=0 means uncapped.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("OMP_NUM_THREADS","1")
import sys, time, threading, numpy as np
from concurrent.futures import ThreadPoolExecutor
D=384; C=2048;  # C is overridden per cell
PW=39; TOP=10; NC=256
P3=3**np.arange(5)
B5=np.array([[(x//3**i)%3 for i in range(5)] for x in range(243)])
V5i=np.where(B5==1,1,np.where(B5==2,-1,0)).astype(np.int8)
V10=np.ascontiguousarray(np.concatenate([np.repeat(V5i,243,0),np.tile(V5i,(243,1))],1))   # 59,049 x 10 int8 pair table
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
class Dev:
    # one request at a time; each request costs a fixed latency plus nbytes at the capped rate
    def __init__(s,mbps,lat_ms): s.bps=mbps*1e6; s.lat=lat_ms/1e3; s.lock=threading.Lock(); s.free=0.0
    def wait(s,nbytes):
        if s.bps<=0: return
        with s.lock:
            st=max(time.perf_counter(),s.free); s.free=st+s.lat+nbytes/s.bps; end=s.free
        d=end-time.perf_counter()
        if d>0: time.sleep(d)
def topk(S,c,C):
    j=np.argpartition(-S,TOP,1)[:,:TOP]; return np.take_along_axis(S,j,1),j+c*C
def sweep(fp,kind,Q,T,N,mbps,lat=0.0,C=2048):
    row=D if kind=="int8" else PW*2; fd=os.open(fp[kind],os.O_RDONLY); dev=Dev(mbps,lat); nch=-(-N//C)
    def chunk(c):
        n=min(C,N-c*C); nb=n*row; dev.wait(nb); buf=os.pread(fd,nb,c*C*row)
        if kind=="int8": X=np.frombuffer(buf,np.int8).reshape(n,D).astype(np.float32)
        else: X=np.take(V10,np.frombuffer(buf,np.uint16).reshape(n,PW),axis=0).reshape(n,PW*10)[:,:D].astype(np.float32)
        return topk(Q@X.T,c,C)
    with ThreadPoolExecutor(T) as ex: parts=list(ex.map(chunk,range(nch)))
    os.close(fd)
    S=np.concatenate([p[0] for p in parts],1); I=np.concatenate([p[1] for p in parts],1)
    o=np.argsort(-S,1,kind="stable")[:,:TOP]; return np.take_along_axis(I,o,1)
def evict(fp):
    for p in fp.values():
        fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
if __name__=="__main__":
    N=int(sys.argv[1]); seed=int(sys.argv[2]); R=int(sys.argv[3])
    RATES=[int(x) for x in sys.argv[4].split(",")]; LATS=[float(x) for x in sys.argv[5].split(",")]; CHS=[int(x) for x in sys.argv[6].split(",")]; T=8
    fp={"int8":"idx_int8.bin","pair":"idx_pair.bin"}
    print(f"numpy {np.__version__} N {N:,} seed {seed} repeats {R} threads {T} rates MB/s {RATES} lat ms {LATS} chunks {CHS}; MemAvailable {open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]} kB; nproc {os.cpu_count()}",flush=True)
    t=time.perf_counter(); Q=build(N,seed,fp); print(f"build {time.perf_counter()-t:.0f}s; bytes int8 {os.path.getsize(fp['int8']):,} pair {os.path.getsize(fp['pair']):,}",flush=True)
    ref=sweep(fp,"int8",Q,1,N,0); assert np.array_equal(sweep(fp,"pair",Q,1,N,0),ref)
    print("cap MB/s | lat ms | chunk rows | int8 median ms (spread) | pair median ms (spread) | int8/pair | top-10")
    for mb in RATES:
      for lat in LATS:
        for C in CHS:
            res={"int8":[],"pair":[]}; ok=True
            for r in range(R):
                for kind in (("int8","pair") if r%2==0 else ("pair","int8")):
                    evict(fp); t=time.perf_counter(); I=sweep(fp,kind,Q,T,N,mb,lat,C); res[kind].append(time.perf_counter()-t); ok&=np.array_equal(I,ref)
            a,b=res["int8"],res["pair"]; ma,mb_=np.median(a),np.median(b)
            print(f"{mb} | {lat:g} | {C} | {ma*1e3:.0f} ({np.ptp(a)*1e3:.0f}) | {mb_*1e3:.0f} ({np.ptp(b)*1e3:.0f}) | {ma/mb_:.2f}x | {'exact' if ok else 'MISMATCH'}",flush=True)
    [os.remove(p) for p in fp.values()]
```

## Results

Cold sweep of 2,000,000 rows by 64 queries, top-10, 8 threads, median ms over 3 repeats (spread = max minus min). Top-10 matched the reference in every cell.

| read cap | latency per request | chunk rows | requests | int8 (768 MB) | pair codes (156 MB) | int8 / pair |
|---|---|---|---|---|---|---|
| 100 MB/s | 0.2 ms | 512 | 3,907 | 8696 (38) | 2559 (6) | 3.40x |
| 100 MB/s | 0.2 ms | 2,048 | 977 | 7920 (9) | 1799 (13) | 4.40x |
| 100 MB/s | 0.2 ms | 8,192 | 245 | 7745 (1) | 1627 (1) | **4.76x** |
| 100 MB/s | 1 ms | 512 | 3,907 | 11796 (46) | 5690 (27) | 2.07x |
| 100 MB/s | 1 ms | 2,048 | 977 | 8701 (51) | 2581 (1) | 3.37x |
| 100 MB/s | 1 ms | 8,192 | 245 | 7939 (2) | 1821 (3) | 4.36x |
| 100 MB/s | 5 ms | 512 | 3,907 | 27425 (22) | 21294 (12) | **1.29x** |
| 100 MB/s | 5 ms | 2,048 | 977 | 12612 (4) | 6491 (2) | 1.94x |
| 100 MB/s | 5 ms | 8,192 | 245 | 8917 (1) | 2798 (1) | **3.19x** |
| 50 MB/s | 0.2 ms | 512 | 3,907 | 16361 (31) | 4102 (11) | 3.99x |
| 50 MB/s | 0.2 ms | 2,048 | 977 | 15599 (5) | 3359 (1) | 4.64x |
| 50 MB/s | 0.2 ms | 8,192 | 245 | 15424 (19) | 3187 (1) | 4.84x |
| 50 MB/s | 1 ms | 512 | 3,907 | 19475 (29) | 7235 (36) | 2.69x |
| 50 MB/s | 1 ms | 2,048 | 977 | 16382 (6) | 4140 (2) | 3.96x |
| 50 MB/s | 1 ms | 8,192 | 245 | 15617 (6) | 3381 (1) | 4.62x |
| 50 MB/s | 5 ms | 512 | 3,907 | 35096 (27) | 22883 (251) | 1.53x |
| 50 MB/s | 5 ms | 2,048 | 977 | 20289 (1) | 8050 (1) | 2.52x |
| 50 MB/s | 5 ms | 8,192 | 245 | 16597 (1) | 4357 (0) | 3.81x |

Per-request latency costs both formats the same fixed time (requests times L), so it eats the pair-code advantage when requests are many and small. At 512-row chunks and 5 ms the fixed cost is about 19.5 s per sweep on either format and the ratio falls to 1.29x at 100 MB/s. At 8,192-row chunks the same latency adds only about 1.2 s, and pair codes stay 3.2x to 4.8x faster.

## Verdict

**Viable, with large chunks.** With 8,192-row chunks, pair codes are 3.19x to 4.84x faster than int8 across every latency and rate tested, with exact top-10. The only cell under 1.5x is 512-row chunks at 5 ms per request on a 100 MB/s device (1.29x). The 2,048-row default from the earlier notes holds 1.94x even in that worst case.

Rule for `night` on SD card, USB or network storage: store the index as pair codes and read 8,192-row chunks (639 KB of pair codes per read). Larger chunks also never hurt int8, which only gained from them here.

Caveats: the device model is in-process. It serializes requests and adds a fixed latency, which is a pessimistic stand-in for cards that queue requests internally. It was not measured on a real Pi or SD card. Synthetic embeddings, one seed, 3 repeats, a shared and noisy machine. Chunk sizes above 8,192 rows were not tested.

## Pair files this wave

- No pair file was visible to this run for the 05:05 wave. JuniorQuant has no commits since 2026-10-29 18:00 MT. In JuniorLLM, the only commit since the 02:55 note is that note itself ([b4d119c](https://github.com/cloudcover95/JuniorLLM/commit/b4d119c)); JuniorOS Builder [8e175d7](https://github.com/cloudcover95/JuniorLLM/commit/8e175d7) (`rails/metro/metro_guard.py`, 02:47 MT) is the only other path touched in the last 6 hours. This note adds one new file under `junior_osai/vault/findings/` only. The ask/night side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

How does `night` behave with a real per-request queue instead of a serialized device? Rerun the 5 ms row with 1, 2, 4 and 8 requests allowed in flight at once, and add 32,768-row chunks, to see whether queue depth or chunk size is the better lever and what peak decode memory the 32,768-row chunks cost at 8 threads.

close: c8b0a92baf18a7b8 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-30_0255)
