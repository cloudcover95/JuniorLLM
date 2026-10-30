---
local: true
date: 2026-10-30
wave: "2026-10-30 02:05 (bench 02:55, ran 03:05 to 03:15)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`night` sweeping the packed ask index)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, night, ask-index, disk-throughput, sd-card, bench]
---

# juniorctl night: throttled-disk crossover, int8 trits vs uint16 pair codes

## Question

The 18:55 note ([[2026-10-29_1855_night-thread-scaling-pair-codes-vs-int8-sweep]]) found that on this box's fast virtio disk the 8-thread cold sweep only reaches 1.28x for pair codes over int8 storage, under the 1.5x bar. It asked for the read rate at which pair codes cross 1.5x. The pair-code index is 4.9x smaller (78 B/row against 384 B/row), so once the sweep is bound by storage rather than by decode, it should approach that ratio. At what read throughput does int8 / pair first reach 1.5x, and how close to 4.9x does it get on slow storage such as the microSD card a Raspberry Pi deck (JuniorHome `docs/DECK_BOM.md`) would boot from?

## Method

Same index as the 18:55 run: 2,000,000 unique rows, 384-dim vectors around 256 topic centres, ternarized with the 0.5 mean|x| absmean rule, written once as int8 trits (768,000,000 bytes) and once as uint16 pair codes (156,000,000 bytes, two 5-trit base-3 groups per code). 64 queries per sweep, top-10, 8 worker threads over 2,048-row chunks, OpenBLAS pinned to 1 thread.

The change is how chunks are read. Each chunk is read with `os.pread` through a shared device model that serves requests one after another at a capped rate R. A reservation clock sits under a lock, and each reader sleeps until its slot ends, then reads. The page cache is evicted with `posix_fadvise(DONTNEED)` before every timed sweep, and "none" means no cap (the real disk). R is 1600, 800, 400, 200, 100 and 50 MB/s. Each cell has 5 repeats with alternating int8 and pair order, and every timed sweep's top-10 indices were checked against a 1-thread uncapped int8 reference.

numpy 2.5.4, 8-vCPU shared box, MemAvailable about 2.5 GB, python3 and numpy only, no network, nothing bound. Index files were deleted afterwards. A 100,000-row smoke run (`python3 bench.py 100000 1 2 0,400,50`) came first. Command, in a scratch folder (not committed):

```
python3 bench.py 2000000 202610300255 5 0,1600,800,400,200,100,50     # N, seed, repeats, read caps in MB/s (0 = uncapped)
```

Full script:

```python
# juniorctl night: throttled-disk crossover, int8 trits (384 B/row) vs uint16 pair codes (78 B/row)
# python3 + numpy only. 8 worker threads, OpenBLAS pinned to 1 thread. Each chunk is read with os.pread through a
# shared device model that serves requests one after another at R MB/s (a reservation clock under a lock; the
# reader sleeps until its slot ends, then preads). Page cache is evicted before every timed sweep. R=0 means uncapped.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("OMP_NUM_THREADS","1")
import sys, time, threading, numpy as np
from concurrent.futures import ThreadPoolExecutor
D=384; C=2048; PW=39; TOP=10; NC=256
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
    def __init__(s,mbps): s.bps=mbps*1e6; s.lock=threading.Lock(); s.free=0.0
    def wait(s,nbytes):
        if s.bps<=0: return
        with s.lock:
            st=max(time.perf_counter(),s.free); s.free=st+nbytes/s.bps; end=s.free
        d=end-time.perf_counter()
        if d>0: time.sleep(d)
def topk(S,c):
    j=np.argpartition(-S,TOP,1)[:,:TOP]; return np.take_along_axis(S,j,1),j+c*C
def sweep(fp,kind,Q,T,N,mbps):
    row=D if kind=="int8" else PW*2; fd=os.open(fp[kind],os.O_RDONLY); dev=Dev(mbps); nch=-(-N//C)
    def chunk(c):
        n=min(C,N-c*C); nb=n*row; dev.wait(nb); buf=os.pread(fd,nb,c*C*row)
        if kind=="int8": X=np.frombuffer(buf,np.int8).reshape(n,D).astype(np.float32)
        else: X=np.take(V10,np.frombuffer(buf,np.uint16).reshape(n,PW),axis=0).reshape(n,PW*10)[:,:D].astype(np.float32)
        return topk(Q@X.T,c)
    with ThreadPoolExecutor(T) as ex: parts=list(ex.map(chunk,range(nch)))
    os.close(fd)
    S=np.concatenate([p[0] for p in parts],1); I=np.concatenate([p[1] for p in parts],1)
    o=np.argsort(-S,1,kind="stable")[:,:TOP]; return np.take_along_axis(I,o,1)
def evict(fp):
    for p in fp.values():
        fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
if __name__=="__main__":
    N=int(sys.argv[1]); seed=int(sys.argv[2]); R=int(sys.argv[3]); RATES=[int(x) for x in sys.argv[4].split(",")]; T=8
    fp={"int8":"idx_int8.bin","pair":"idx_pair.bin"}
    print(f"numpy {np.__version__} N {N:,} seed {seed} repeats {R} threads {T} rates MB/s {RATES}; MemAvailable {open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]} kB; nproc {os.cpu_count()}",flush=True)
    t=time.perf_counter(); Q=build(N,seed,fp); print(f"build {time.perf_counter()-t:.0f}s; bytes int8 {os.path.getsize(fp['int8']):,} pair {os.path.getsize(fp['pair']):,}",flush=True)
    ref=sweep(fp,"int8",Q,1,N,0); assert np.array_equal(sweep(fp,"pair",Q,1,N,0),ref)
    print("cap MB/s | int8 median ms (spread) | pair median ms (spread) | int8/pair | int8 eff MB/s | pair eff MB/s | top-10")
    for mb in RATES:
        res={"int8":[],"pair":[]}; ok=True
        for r in range(R):
            for kind in (("int8","pair") if r%2==0 else ("pair","int8")):
                evict(fp); t=time.perf_counter(); I=sweep(fp,kind,Q,T,N,mb); res[kind].append(time.perf_counter()-t); ok&=np.array_equal(I,ref)
        a,b=res["int8"],res["pair"]; ma,mb_=np.median(a),np.median(b)
        print(f"{mb if mb else 'none'} | {ma*1e3:.0f} ({np.ptp(a)*1e3:.0f}) | {mb_*1e3:.0f} ({np.ptp(b)*1e3:.0f}) | {ma/mb_:.2f}x | {os.path.getsize(fp['int8'])/ma/1e6:.0f} | {os.path.getsize(fp['pair'])/mb_/1e6:.0f} | {'exact' if ok else 'MISMATCH'}",flush=True)
    [os.remove(p) for p in fp.values()]
```

## Results

Cold sweep of 2,000,000 rows by 64 queries, top-10, 8 threads, median ms over 5 repeats (spread = max minus min). Effective MB/s is file bytes divided by the median sweep time.

| read cap | int8 (768 MB) | pair codes (156 MB) | int8 / pair | int8 eff MB/s | pair eff MB/s | top-10 vs reference |
|---|---|---|---|---|---|---|
| none | 394 (26) | 306 (22) | 1.29x | 1947 | 510 | exact |
| 1600 MB/s | 537 (23) | 333 (38) | **1.61x** | 1430 | 469 | exact |
| 800 MB/s | 1006 (10) | 342 (38) | 2.94x | 763 | 456 | exact |
| 400 MB/s | 1966 (14) | 443 (9) | 4.44x | 391 | 352 | exact |
| 200 MB/s | 3887 (7) | 832 (14) | 4.67x | 198 | 187 | exact |
| 100 MB/s | 7729 (7) | 1613 (20) | 4.79x | 99 | 97 | exact |
| 50 MB/s | 15407 (5) | 3167 (8) | **4.87x** | 50 | 49 | exact |

The uncapped row reproduces the 18:55 result (1.29x there, 1.28x at 8 threads cold). Pair codes are compute-bound at about 300 to 340 ms down to an 800 MB/s cap. Below about 450 MB/s both paths are storage-bound, and the ratio approaches the 4.92x byte ratio.

## Verdict

**Viable, on storage at or below about 1.6 GB/s.** The crossover to 1.5x sits between the uncapped disk (about 1.9 GB/s effective, 1.29x) and a 1.6 GB/s cap (1.61x). At 400 MB/s and below, pair codes are 4.4x to 4.9x faster per nightly sweep with exact top-10, because they read 4.9x fewer bytes and the decode finishes well inside the read time. A microSD card on a UHS-I bus is capped at 104 MB/s by the bus spec, which is the 100 MB/s row here (4.79x, 7.7 s against 1.6 s per sweep).

This refines the `night` rule from the 10-27 to 10-29 notes. Use int8 storage when the index sits in the page cache or on fast NVMe or virtio storage. Use pair codes when the sweep reads from storage slower than about 1.6 GB/s, such as an SD card, USB stick or network volume, as well as when int8 cannot fit at all.

Caveats: the read cap is an in-process model of a device that serves one request at a time at a fixed rate. It does not model SD card seek latency, small-read overhead or thermal throttling, and it was not measured on a real Pi or SD card. Synthetic embeddings, one seed, a shared and noisy machine. The 1.6 GB/s crossover is bracketed by two points, not pinned.

## Pair files this wave

- No pair file was visible to this run for the 02:05 wave. JuniorQuant, BitNet-mlx and JuniorStock have no commits since 2026-10-29 09:00 MT. In JuniorLLM, the only commit since the 18:55 note is JuniorOS Builder [8e175d7](https://github.com/cloudcover95/JuniorLLM/commit/8e175d7), which added `rails/metro/metro_guard.py`. This note adds one new file under `junior_osai/vault/findings/` only and touches no path from the last 6 hours. The ask/night side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Check the device model against small-read latency. Rerun the 100 MB/s and 50 MB/s rows with a fixed per-request delay added (0.2, 1 and 5 ms per chunk read, which brackets SD card random-read latency), and with chunk sizes of 512, 2,048 and 8,192 rows. Does the pair-code advantage stay above 1.5x once per-request cost dominates, and which chunk size `night` should use on an SD card?

close: 4ee2e7d908bbc83a (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-29_1855)
