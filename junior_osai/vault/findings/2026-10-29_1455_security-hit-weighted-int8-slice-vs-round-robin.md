---
local: true
date: 2026-10-29
wave: "2026-10-29 14:05 (bench 14:55, ran 15:04 to 15:15)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`security` re-attesting the packed ask index that `night` sweeps as int8)"
verdict: inconclusive
tags: [osai, vault, findings, ternary, pair-codes, int8, merkle, sha256, juniorctl, security, night, attestation, scheduling, bench]
---

# juniorctl security: hit-weighted int8 decode-compare slice vs `c % 8` round-robin at the same nightly read budget

## Question

The 10:55 note ([[2026-10-29_1055_security-rotating-int8-decode-slice-vs-int8-leaves]]) made a rotating 1/8 decode-compare of the int8 copy against the sha256-verified ternary pair-code leaves the nightly default (2.86x warm, 2.08x cold, int8-only change caught within 7 nights). Its next question: at the same 1/8 int8 read budget, can the slice be weighted toward chunks that the int8 `night` sweep actually returns in its top results, and does that cut the expected nights-to-detect for changes that would matter to `ask` answers by at least 1.5x? No new roadmap item or pair file was visible to this run (JuniorLLM has not moved since f5a65a4), so this stays on the ask/night/security line.

## Method

Same pair-code encoder, 0.5 mean|x| absmean ternarization, 2,048-row chunks and Merkle re-attest as the 10:55 run (sha256 every pair leaf, decode-compare dirty chunks plus the night's scheduled clean chunks against the int8 copy). Changes from 10:55: all 2,000,000 rows are unique (no 50k tiling, so top-k hits are real), then a 1% append gives 2,020,000 rows, 987 chunks, 11 dirty, 976 clean, budget 122 clean chunks per night (= ceil(976 / 8), the 10:55 k = 8 read).

Two corpus layouts over 256 topic centres: **shuffled** (topic drawn per row) and **topic-runs** (consecutive rows share a topic, mean run 4,096 rows, as when a folder is ingested in one go). Queries come from a Zipf(1.1) topic popularity. Log A (512 queries, "yesterday") is run through an int8 `night` sweep (float32 matmul over a memmap of the int8 copy, top-10 rows per query) and its per-chunk hit counts build the schedules; log B (512 fresh queries from the same popularity) is held out and used to score them, so the result isn't fitted to the queries it is judged on.

Schedules, each giving every clean chunk a period and offset:

- **round-robin `c % 8` (10:55)**: period 8 for every chunk.
- **hot-half nightly + round-robin rest**: the 61 hottest chunks every night, the rest round-robin in the other 61 slots (period 15).
- **sqrt-weighted periods (cap 16)**: power-of-two periods from 1 to 16, greedily halved where hit weight x period squared is largest (the square-root rule), offsets balanced so no night reads more than 122 clean chunks.

Expected nights-to-detect for a change made on a uniformly random night is (period - 1) / 2 for its chunk; "hit-weighted" weights chunks by log-B top-10 hits, "uniform" weights every clean chunk equally. Five repeats of warm (one untimed run first) and cold (`posix_fadvise(DONTNEED)`) re-attests per schedule; every timed run returned "agree". Then six real int8-only tamper trials per layout (one trit flipped in a row of a chunk drawn by log-B hit weight, random start night), running real nightly re-attests until caught, bytes restored and all schedules re-checked after each. numpy 2.5.4, 8-vCPU shared box, MemAvailable about 6.5 GB, python3 and numpy only, no network, nothing bound; index files deleted after each layout. A 100,000-row, 2-repeat smoke run (`python3 bench.py 100000 1 2 64`) came first. Command, in a scratch folder (not committed):

```
python3 bench.py 2000000 202610291455 5 512     # N, seed, repeats, queries per log
```

Full script:

```python
# juniorctl security/night: hit-weighted int8 decode-compare slice vs c % 8 round-robin at the same nightly int8 read budget
# (python3 + numpy only). Weights come from yesterday's int8 `night` sweep top-10 hits (query log A); scored on a held-out log B.
import os, sys, time, heapq, hashlib, numpy as np
D=384; C=2048; PW=39; K=8; TOP=10; NC=256
P3=3**np.arange(5)
B5=np.array([[(x//3**i)%3 for i in range(5)] for x in range(243)])
V5i=np.where(B5==1,1,np.where(B5==2,-1,0)).astype(np.int8)
V10=np.concatenate([np.repeat(V5i,243,0),np.tile(V5i,(243,1))],1)
def enc_pair(T):
    code=np.where(T==1,1,np.where(T==-1,2,0)).astype(np.uint16); n=len(T)
    b=(np.concatenate([code,np.zeros((n,1),np.uint16)],1).reshape(n,77,5)*P3).sum(2)
    bp=np.concatenate([b,np.zeros((n,1),np.uint16)],1); return (bp[:,0::2]*243+bp[:,1::2]).astype(np.uint16)
dec_pair=lambda r:V10[r].reshape(len(r),390)[:,:D]
RB={"pair":PW*2,"int8":D}
def tern(X):
    X/=np.linalg.norm(X,axis=1,keepdims=True); a=np.abs(X)
    return np.where(a>0.5*a.mean(1,keepdims=True),np.sign(X),0).astype(np.int8)
def topics(rng,n,layout):
    if layout=="shuffled": return rng.integers(0,NC,n)
    out=[];m=0                                   # topic runs: consecutive docs share a topic (folder-style ingest), mean run 4096 rows
    while m<n: L=int(rng.geometric(1/4096)); out.append(np.full(L,rng.integers(NC))); m+=L
    return np.concatenate(out)[:n]
def write(rng,Cc,top,fp,mode):
    with open(fp["int8"],mode) as fi, open(fp["pair"],mode) as fq:
        for s in range(0,len(top),65536):
            t=top[s:s+65536]; T=tern(Cc[t]+0.9*rng.standard_normal((len(t),D)).astype(np.float32))
            T.tofile(fi); enc_pair(T).tofile(fq)
        for f in (fi,fq): f.flush(); os.fsync(f.fileno())
class Idx:
    def __init__(s,fp): s.fp=fp; s.fd={k:os.open(fp[k],os.O_RDWR) for k in fp}
    def rows(s): return os.fstat(s.fd["pair"]).st_size//RB["pair"]
    def nch(s): return -(-s.rows()//C)
    def raw(s,k,c):
        n=s.rows(); a=c*C; b=min(n,a+C); return os.pread(s.fd[k],(b-a)*RB[k],a*RB[k])
    def close(s): [os.close(v) for v in s.fd.values()]
sha=lambda b:hashlib.sha256(b).digest()
def cmp(pb,ib): return np.array_equal(dec_pair(np.frombuffer(pb,np.uint16).reshape(-1,PW)),np.frombuffer(ib,np.int8).reshape(-1,D))
def root(L):
    L=list(L)
    while len(L)>1:
        if len(L)&1: L.append(L[-1])
        L=[hashlib.sha256(b"\x01"+L[i]+L[i+1]).digest() for i in range(0,len(L),2)]
    return L[0]
def attest(ix,st,dirty,S):           # 10:55 shape: sha256 every pair leaf; decode-compare dirty + scheduled clean chunks S
    ok=True; L=[]
    for c in range(ix.nch()):
        pb=ix.raw("pair",c); l=sha(pb); L.append(l)
        if c in dirty or c in S: ok&=cmp(pb,ix.raw("int8",c))
        if c not in dirty: ok&=(l==st[c])
    root(L); return ok
def sweep(ix,Q):                     # int8 `night` sweep: top-10 rows per query, returns per-chunk hit counts
    mm=np.memmap(ix.fp["int8"],np.int8,"r",shape=(ix.rows(),D)); bs=np.full((len(Q),TOP),-np.inf,np.float32); bi=np.zeros((len(Q),TOP),np.int64)
    for s in range(0,len(mm),65536):
        S=Q@mm[s:s+65536].astype(np.float32).T; j=np.argpartition(-S,TOP,1)[:,:TOP]
        cs=np.concatenate([bs,np.take_along_axis(S,j,1)],1); ci=np.concatenate([bi,j+s],1); o=np.argsort(-cs,1)[:,:TOP]
        bs=np.take_along_axis(cs,o,1); bi=np.take_along_axis(ci,o,1)
    del mm; return np.bincount((bi//C).ravel(),minlength=ix.nch())
# schedules over clean chunks 0..fd0-1: each chunk c gets (period p_c, offset o_c), checked on night n iff n % p_c == o_c
def sched_rr(fd0,w,B): return {c:(K,c%K) for c in range(fd0)}
def sched_hot(fd0,w,B):              # half the budget checks the hottest chunks every night, the rest round-robins the remainder
    order=np.argsort(-w[:fd0],kind="stable"); H=[int(c) for c in order[:B//2] if w[c]>0]; R=[int(c) for c in order if int(c) not in set(H)]
    P=-(-len(R)//(B-len(H))); s={c:(1,0) for c in H}; s.update({c:(P,i%P) for i,c in enumerate(sorted(R))}); return s
def sched_sqrt(fd0,w,B,cap=16):      # greedy power-of-two periods, gain/cost = w*p^2/4 (sqrt-rule), cap 16 nights, offsets balanced
    p={c:cap for c in range(fd0)}; cost=fd0/cap; h=[(-(w[c]+1e-3)*cap*cap/4,c) for c in range(fd0)]; heapq.heapify(h)
    while h:
        g,c=heapq.heappop(h); q=p[c]
        if q==1 or cost+1/q>B: continue
        p[c]=q//2; cost+=1/q
        if p[c]>1: heapq.heappush(h,(-(w[c]+1e-3)*p[c]*p[c]/4,c))
    s={}; load=np.zeros(cap)
    for c in sorted(range(fd0),key=lambda c:(p[c],c)):
        q=p[c]; o=int(np.argmin([load[i::q].max() if q<cap else load[i] for i in range(q)])); load[o::q]+=1; s[c]=(q,o)
    return s
S_of=lambda s,n:{c for c,(p,o) in s.items() if n%p==o}
def lat(s,w):                        # expected / worst nights-to-detect for a change made on a uniform random night
    m=np.array([(s[c][0]-1)/2 for c in range(len(s))]); return (m*w).sum()/w.sum(), m.mean(), max(p for p,_ in s.values())-1
def evict(fp):
    for p in fp.values():
        fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
if __name__=="__main__":
    N=int(sys.argv[1]); seed=int(sys.argv[2]); R=int(sys.argv[3]); NQ=int(sys.argv[4]); APP=N//100
    print(f"numpy {np.__version__} N {N:,} seed {seed} repeats {R} queries/log {NQ}; MemAvailable {open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]} kB; nproc {os.cpu_count()}",flush=True)
    for layout in ("shuffled","topic-runs"):
        rng=np.random.default_rng(seed); Cc=rng.standard_normal((NC,D)).astype(np.float32)
        fp={"pair":f"idx_pair_{layout}.bin","int8":f"idx_int8_{layout}.bin"}
        t0=time.perf_counter(); write(rng,Cc,topics(rng,N,layout),fp,"wb"); ix=Idx(fp); fd0=N//C
        st={c:sha(ix.raw("pair",c)) for c in range(fd0)}
        write(rng,Cc,topics(rng,APP,layout),fp,"ab"); dirty=set(range(fd0,ix.nch())); B=-(-fd0//K)
        pop=1/np.arange(1,NC+1)**1.1; pop/=pop.sum(); perm=rng.permutation(NC)
        def qlog():
            Q=Cc[perm[rng.choice(NC,NQ,p=pop)]]+0.9*rng.standard_normal((NQ,D)).astype(np.float32); return Q/np.linalg.norm(Q,axis=1,keepdims=True)
        QA,QB=qlog(),qlog()
        t=time.perf_counter(); wA=sweep(ix,QA); tsw=time.perf_counter()-t; wB=sweep(ix,QB)
        t=time.perf_counter(); scheds={"round-robin c % 8 (10:55)":sched_rr(fd0,wA,B),"hot-half nightly + round-robin rest":sched_hot(fd0,wA,B),"sqrt-weighted periods (cap 16)":sched_sqrt(fd0,wA,B)}; tsc=time.perf_counter()-t
        nz=(wA[:fd0]>0).sum(); top=np.sort(wA[:fd0])[::-1]; share=top[:B//2].sum()/wA[:fd0].sum()
        print(f"\n[{layout}] rows {ix.rows():,} chunks {ix.nch()} dirty {len(dirty)} budget {B} clean chunks/night; build {time.perf_counter()-t0:.0f}s; sweep {tsw:.1f}s; schedules built in {tsc*1e3:.0f} ms",flush=True)
        print(f"[{layout}] log A hits: {nz}/{fd0} clean chunks hit, hottest {B//2} hold {share:.0%} of hits; log A/B hit-count overlap of top {B//2}: {len(set(np.argsort(-wA[:fd0])[:B//2])&set(np.argsort(-wB[:fd0])[:B//2]))}/{B//2}",flush=True)
        print("layout | schedule | max chunks/night | hit-weighted E[nights] log A (in-sample) | log B (held-out) | uniform E[nights] | worst nights | rr/sched on log B")
        base=lat(scheds["round-robin c % 8 (10:55)"],wB[:fd0]+0.0)[0]
        for n,s in scheds.items():
            a=lat(s,wA[:fd0]+0.0); b=lat(s,wB[:fd0]+0.0); mx=max(len(S_of(s,x)) for x in range(16))
            print(f"{layout} | {n} | {mx} | {a[0]:.2f} | {b[0]:.2f} | {b[1]:.2f} | {b[2]} | {base/b[0]:.2f}x",flush=True)
        res={}
        for r in range(R):
            for mode in ("warm","cold"):
                for n,s in scheds.items():
                    S=S_of(s,r)
                    if mode=="cold": evict(fp)
                    else: attest(ix,st,dirty,S)
                    t=time.perf_counter(); ok=attest(ix,st,dirty,S); dt=time.perf_counter()-t
                    assert ok,(layout,mode,n); res.setdefault((mode,n),[]).append(dt)
        print("layout | mode | schedule | median ms (spread)")
        for mode in ("warm","cold"):
            for n in scheds: v=res[(mode,n)]; print(f"{layout} | {mode} | {n} | {np.median(v)*1e3:.1f} ({np.ptp(v)*1e3:.1f})",flush=True)
        # real tamper: int8-only trit flip in a row that log B's sweep returns, chunk drawn by log-B hit weight; nightly re-attest until caught
        trng=np.random.default_rng(seed+1); pw=wB[:fd0]/wB[:fd0].sum(); print("layout | tamper chunk (log-B hits) | start night | nights to detect per schedule")
        for trial in range(6):
            c=int(trng.choice(fd0,p=pw)); rr=c*C+int(trng.integers(C)); col=int(trng.integers(D))
            old=os.pread(ix.fd["int8"],D,rr*D); row=np.frombuffer(old,np.int8).copy(); row[col]=-1 if row[col]==1 else 1; os.pwrite(ix.fd["int8"],row.tobytes(),rr*D)
            start=int(trng.integers(64)); out=[]
            for n,s in scheds.items():
                got="never"
                for d in range(17):
                    if not attest(ix,st,dirty,S_of(s,start+d)): got=str(d); break
                out.append(f"{n}: {got}")
            os.pwrite(ix.fd["int8"],old,rr*D); assert all(attest(ix,st,dirty,S_of(s,start)) for s in scheds.values())
            print(f"{layout} | {c} ({wB[c]}) | {start} | "+"; ".join(out),flush=True)
        ix.close(); [os.remove(p) for p in fp.values()]
```

## Results

Detection latency at the same 122-chunk nightly read (max chunks read on any night was 122 for every schedule):

| layout | schedule | hit-weighted E[nights], log A (in-sample) | **hit-weighted E[nights], log B (held-out)** | uniform E[nights] | worst nights | round-robin / schedule, log B |
|---|---|---|---|---|---|---|
| shuffled | round-robin `c % 8` (10:55) | 3.50 | 3.50 | 3.50 | 7 | 1.00x |
| shuffled | hot-half nightly + round-robin rest | 6.06 | 6.31 | 6.56 | 14 | 0.55x |
| shuffled | sqrt-weighted periods (cap 16) | 3.38 | 3.75 | 3.92 | 15 | 0.93x |
| topic-runs | round-robin `c % 8` (10:55) | 3.50 | 3.50 | 3.50 | 7 | 1.00x |
| topic-runs | hot-half nightly + round-robin rest | 2.74 | 3.26 | 6.56 | 14 | 1.07x |
| topic-runs | **sqrt-weighted periods (cap 16)** | 1.38 | **2.16** | 5.29 | 15 | **1.62x** |

Hit concentration: shuffled, 966 of 976 clean chunks were hit and the hottest 61 held 13% of log-A hits (only 18 of those 61 were also log B's top 61); topic-runs, 635 of 976 hit, hottest 61 held 61% (46 of 61 shared with log B). The log-A sweep took 7.6 s (shuffled) and 7.9 s (topic-runs), and building all three schedules from its hit counts took 11 to 12 ms.

Nightly re-attest cost, 2,020,000 rows, 5 repeats (the budget is the same, so these should match):

| layout | mode | round-robin `c % 8` | hot-half + round-robin | sqrt-weighted |
|---|---|---|---|---|
| shuffled | warm | 272.5 ms (38.1) | 282.7 ms (16.8) | 276.6 ms (75.2) |
| shuffled | cold | 436.9 ms (116.5) | 426.2 ms (118.2) | 460.8 ms (79.0) |
| topic-runs | warm | 274.1 ms (55.6) | 277.7 ms (33.6) | 269.5 ms (28.1) |
| topic-runs | cold | 474.2 ms (75.0) | 403.8 ms (58.1) | 432.1 ms (97.9) |

Real int8-only tamper trials, nights to detect (0 = the night of the change), round-robin / hot-half / sqrt-weighted, chunk (log-B hits):

| layout | trials |
|---|---|
| shuffled | 839 (5): 2/2/2; 918 (10): 3/14/0; 308 (8): 7/11/5; 364 (2): 0/14/5; 609 (4): 2/14/4; 175 (6): 6/7/0 |
| topic-runs | 790 (3): 1/2/7; 872 (109): 5/0/0; 435 (27): 6/0/0; 484 (269): 0/0/0; 572 (1): 5/0/4; 225 (56): 0/0/0 |

Every observed value is inside its schedule's period bound.

## Verdict

**Inconclusive; keep `c % 8` as the default.** The cost side is a wash by construction: every schedule reads the same 122 clean int8 chunks a night and runs in 270 to 280 ms warm and 400 to 475 ms cold, so there is no speed or memory gain to claim. On detection, the square-root-weighted schedule cuts the held-out hit-weighted expected latency 1.62x (2.16 vs 3.50 nights) only when topics arrive in runs, so that `ask` hits pile into a few hundred chunks. On a shuffled layout, where hits spread over almost every chunk and yesterday's hot set barely predicts today's (18 of 61), it is 0.93x, and the hot-half schedule is a net loss on both layouts held-out (0.55x and 1.07x). Either weighted schedule also roughly doubles the worst case for a cold chunk from 7 to 14 or 15 nights, and raises the uniform expected latency to 5.3 to 6.6 nights. So the gain depends on how the real index is laid out, and it trades away the clean 7-night bound. This is a scheduling result, not a ternary one: the pair-code anchor makes any slice policy safe, but it doesn't make weighting cheaper.

Caveats: synthetic topic layout and Zipf query popularity, one seed, 512-query logs; the real `night` sweep's hit distribution was not available to this run. Six tamper trials per layout are a sanity check; the expected latencies are computed exactly from the periods.

## Pair files this wave

- No pair file was visible to this run for the 14:05 wave. JuniorLLM has not moved since the 10:55 note ([f5a65a4](https://github.com/cloudcover95/JuniorLLM/commit/f5a65a4)), and this note adds one new file under `junior_osai/vault/findings/` only. The ask/security side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

How clustered is the real index? Measure, from a local copy of the packed ask index's chunk order (or from the `night` sweep's own top-k output if one is on disk), what share of top-10 hits the hottest 1/16 of chunks hold, and check whether it clears the roughly 60% concentration at which the square-root schedule reached 1.5x here. If no real hit data is reachable locally, switch lanes: compare the pair-code (78 B/row) vs int8 (384 B/row) `night` sweep itself at 2M unique rows, decode-then-matmul vs int8 matmul, warm and cold.

close: 0a3f0cea76942b67 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-29_1055)
