---
local: true
date: 2026-10-30
wave: "2026-10-30 10:05 (bench 10:55, ran 11:06 to 11:16)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`night` sweeping the packed ask index)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, night, ask-index, sd-card, queue-depth, chunk-size, memory, bench]
---

# juniorctl night: requests in flight and chunk size at 5 ms per request, int8 trits vs uint16 pair codes

## Question

The 05:55 note ([[2026-10-30_0555_night-read-latency-chunk-size-pair-codes-vs-int8]]) found that a 5 ms per-request cost on a serialized device cuts the pair-code advantage to 1.29x at 512-row chunks, while 8,192-row chunks keep it above 3x. Real cards and USB sticks can overlap several requests. If up to 1, 2, 4 or 8 requests can be in flight, is queue depth or chunk size the better lever, and does going up to 32,768-row chunks help enough to pay for its extra decode memory at 8 threads?

## Method

Same index and sweep as the 02:55 and 05:55 runs: 2,000,000 unique rows, 384-dim, ternarized with the 0.5 mean|x| absmean rule, stored as int8 trits (768,000,000 bytes) and as uint16 pair codes (156,000,000 bytes). 64 queries, top-10, 8 worker threads, OpenBLAS pinned to 1 thread, page cache evicted before every timed sweep.

The one change is the device model. Up to K requests may be in flight at once: each waits the fixed 5 ms latency concurrently with the others (a K-slot semaphore), and then its bytes stream one request at a time at 100 MB/s. K = 1 reproduces the 05:55 serialized device. The grid is K of 1, 2, 4 and 8 and chunk sizes of 512, 8,192 and 32,768 rows (3,907, 245 and 62 requests per sweep on either format). Each cell has 3 repeats with alternating int8 and pair order, and every timed sweep's top-10 was checked against a 1-thread uncapped int8 reference.

Peak decode memory is the tracemalloc peak (numpy buffers included) during one uncapped 8-thread sweep per format and chunk size. It counts Python and numpy allocations only, not the page cache or interpreter baseline.

numpy 2.5.4, 8-vCPU shared box, MemAvailable about 6.6 GB at start, python3 and numpy only, no network, nothing bound. Index files were deleted afterwards. A 100,000-row smoke run (`python3 bench.py 100000 1 1 100 5 1,8 512,32768`) came first. Command, in a scratch folder (not committed):

```
python3 bench.py 2000000 202610301055 3 100 5 1,2,4,8 512,8192,32768     # N, seed, repeats, read cap MB/s, latency ms, requests in flight, chunk rows
```

Full script:

```python
# juniorctl night: requests in flight (queue depth) x chunk size at 5 ms per request, int8 trits (384 B/row) vs uint16 pair codes (78 B/row)
# python3 + numpy only. 8 worker threads, OpenBLAS pinned to 1 thread. Each chunk is read with os.pread through a
# shared device model: up to K requests in flight each wait the fixed latency L concurrently (a K-slot semaphore),
# then their bytes are served one after another at R MB/s (a reservation clock under a lock); the reader sleeps until
# its slot ends, then preads. Page cache is evicted before every timed sweep. R=0 means uncapped.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("OMP_NUM_THREADS","1")
import sys, time, threading, tracemalloc, numpy as np
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
    # K requests in flight share the fixed latency; bytes stream one request at a time at the capped rate
    def __init__(s,mbps,lat_ms,K=1): s.bps=mbps*1e6; s.lat=lat_ms/1e3; s.lock=threading.Lock(); s.free=0.0; s.q=threading.Semaphore(K)
    def wait(s,nbytes):
        if s.bps<=0: return
        with s.q:
            time.sleep(s.lat)
            with s.lock:
                st=max(time.perf_counter(),s.free); s.free=st+nbytes/s.bps; end=s.free
            d=end-time.perf_counter()
            if d>0: time.sleep(d)
def topk(S,c,C):
    j=np.argpartition(-S,TOP,1)[:,:TOP]; return np.take_along_axis(S,j,1),j+c*C
def sweep(fp,kind,Q,T,N,mbps,lat=0.0,C=2048,K=1):
    row=D if kind=="int8" else PW*2; fd=os.open(fp[kind],os.O_RDONLY); dev=Dev(mbps,lat,K); nch=-(-N//C)
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
    N=int(sys.argv[1]); seed=int(sys.argv[2]); R=int(sys.argv[3]); mb=int(sys.argv[4]); lat=float(sys.argv[5])
    KS=[int(x) for x in sys.argv[6].split(",")]; CHS=[int(x) for x in sys.argv[7].split(",")]; T=8
    fp={"int8":"idx_int8.bin","pair":"idx_pair.bin"}
    print(f"numpy {np.__version__} N {N:,} seed {seed} repeats {R} threads {T} rate {mb} MB/s lat {lat:g} ms depths {KS} chunks {CHS}; MemAvailable {open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]} kB; nproc {os.cpu_count()}",flush=True)
    t=time.perf_counter(); Q=build(N,seed,fp); print(f"build {time.perf_counter()-t:.0f}s; bytes int8 {os.path.getsize(fp['int8']):,} pair {os.path.getsize(fp['pair']):,}",flush=True)
    ref=sweep(fp,"int8",Q,1,N,0); assert np.array_equal(sweep(fp,"pair",Q,1,N,0),ref)
    print("chunk rows | peak traced MB int8 | peak traced MB pair (uncapped 8-thread sweep, tracemalloc)")
    for C in CHS:
        pk={}
        for kind in ("int8","pair"):
            tracemalloc.start(); sweep(fp,kind,Q,T,N,0,0,C); pk[kind]=tracemalloc.get_traced_memory()[1]/1e6; tracemalloc.stop()
        print(f"{C} | {pk['int8']:.1f} | {pk['pair']:.1f}",flush=True)
    print("in flight | chunk rows | int8 median ms (spread) | pair median ms (spread) | int8/pair | top-10")
    for K in KS:
        for C in CHS:
            res={"int8":[],"pair":[]}; ok=True
            for r in range(R):
                for kind in (("int8","pair") if r%2==0 else ("pair","int8")):
                    evict(fp); t=time.perf_counter(); I=sweep(fp,kind,Q,T,N,mb,lat,C,K); res[kind].append(time.perf_counter()-t); ok&=np.array_equal(I,ref)
            a,b=res["int8"],res["pair"]; ma,mb_=np.median(a),np.median(b)
            print(f"{K} | {C} | {ma*1e3:.0f} ({np.ptp(a)*1e3:.0f}) | {mb_*1e3:.0f} ({np.ptp(b)*1e3:.0f}) | {ma/mb_:.2f}x | {'exact' if ok else 'MISMATCH'}",flush=True)
    [os.remove(p) for p in fp.values()]
```

## Results

Cold sweep of 2,000,000 rows by 64 queries, top-10, 8 threads, 100 MB/s, 5 ms per request, median ms over 3 repeats (spread = max minus min). Top-10 matched the reference in every cell.

| in flight | chunk rows | requests | int8 (768 MB) | pair codes (156 MB) | int8 / pair |
|---|---|---|---|---|---|
| 1 | 512 | 3,907 | 28956 (203) | 22667 (222) | **1.28x** |
| 1 | 8,192 | 245 | 9020 (10) | 2893 (31) | 3.12x |
| 1 | 32,768 | 62 | 8051 (9) | 1934 (11) | 4.16x |
| 2 | 512 | 3,907 | 14613 (94) | 11452 (36) | 1.28x |
| 2 | 8,192 | 245 | 7701 (3) | 1584 (1) | **4.86x** |
| 2 | 32,768 | 62 | 7727 (15) | 1610 (15) | 4.80x |
| 4 | 512 | 3,907 | 7927 (60) | 5836 (79) | 1.36x |
| 4 | 8,192 | 245 | 7701 (3) | 1583 (2) | 4.86x |
| 4 | 32,768 | 62 | 7723 (3) | 1606 (11) | 4.81x |
| 8 | 512 | 3,907 | 7964 (102) | 3384 (36) | 2.35x |
| 8 | 8,192 | 245 | 7704 (3) | 1584 (2) | 4.86x |
| 8 | 32,768 | 62 | 7731 (2) | 1605 (3) | 4.82x |

Peak traced decode memory, uncapped 8-thread sweep:

| chunk rows | int8 peak MB | pair codes peak MB |
|---|---|---|
| 512 | 91.5 | 91.4 |
| 8,192 | 179.7 | 158.8 |
| 32,768 | 713.2 | 624.6 |

With 512-row chunks the sweep is latency-bound, so extra queue depth speeds both formats by about the same factor and the ratio stays at 1.28x to 1.36x until 8 requests are in flight. Only at K = 8 does int8 hit the 100 MB/s byte limit (7.96 s, near 768 MB / 100 MB/s) while pair codes are still latency-bound, which lifts the ratio to 2.35x. With 8,192-row chunks, two requests in flight are enough to hide the latency completely: both formats reach their byte limit (7.70 s and 1.58 s) and pair codes are 4.86x faster. 32,768-row chunks only help on a strictly serialized device (4.16x vs 3.12x at K = 1); from K = 2 up they are no faster than 8,192 rows and need about 4x the decode memory (625 to 713 MB vs 159 to 180 MB).

## Verdict

**Viable.** Chunk size is the better lever. With 8,192-row chunks and at least 2 requests in flight, pair codes sweep the 2M-row index 4.86x faster than int8 at 5 ms per request, with exact top-10 and 159 MB peak decode memory. Queue depth alone does not rescue 512-row chunks (1.28x to 1.36x up to K = 4). 32,768-row chunks are worth it only on a device that cannot overlap any requests, and cost about 625 MB of decode memory at 8 threads, which is too much next to bitnetd's 2 GB MemoryMax.

Rule for `night`: keep the 05:55 rule of pair codes with 8,192-row chunks and keep 2 or more reads outstanding (the 8 worker threads already do this). Do not raise the chunk size to 32,768 rows.

Caveats: the device model is in-process (latency overlaps across K slots, bytes stream serially) and was not measured on a real Pi, SD card or USB stick. One rate (100 MB/s), one latency (5 ms), synthetic embeddings, one seed, 3 repeats, a shared machine. tracemalloc slows the uncapped sweeps it watches, so it was used only for the memory table.

## Pair files this wave

- No pair file was produced for the 10:05 wave: Quant Chief assigned "no gap" to both pairs (roadmap blobs unchanged). In JuniorLLM, the only commit in the last 6 hours is the 05:55 note itself ([50bd4c3](https://github.com/cloudcover95/JuniorLLM/commit/50bd4c3)), so no Trit Pack, Handshake or LLC path was touched. This note adds one new file under `junior_osai/vault/findings/` only. The ask/night side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Is the night sweep still I/O-bound once reads are hidden? Rerun the 8,192-row, K = 2 cell at 400 MB/s and 1.6 GB/s with 2, 4 and 8 worker threads to find where pair-code decode (the 59,049 x 10 table gather) becomes the bottleneck and the ratio starts to fall toward the 1.29x uncapped figure from 02:55.

close: eb00331d327a5137 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-30_0555)
