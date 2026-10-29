---
local: true
date: 2026-10-29
wave: "2026-10-29 10:05 (bench 10:55, ran 11:04 to 11:10)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`security` re-attesting the packed ask index that `night` sweeps as int8)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, merkle, sha256, juniorctl, security, night, attestation, tamper, rotation, bench]
---

# juniorctl security: rotating 1/k decode-compare of the int8 copy against verified pair leaves vs per-chunk int8 leaves

## Question

The 05:55 note ([[2026-10-29_0555_security-int8-leaf-sha256-vs-blake2b-pair-reencode-2bit-pack]]) showed the int8 leaf can't be made cheaper than raw sha256, so `all_pair+int8_leaves` stays about 700 ms warm at 2M rows, mostly spent hashing the 776 MB int8 copy. Its next question: drop the stored int8 leaves, keep sha256 on every ternary pair-code leaf (the 78 B/row authoritative copy), and instead decode-compare a rotating 1/k slice of clean chunks each night against the int8 copy, so the int8 copy is fully covered every k nights. Is any k at least 1.5x cheaper per night, warm and cold, and what is the detection-latency bound for an int8-only change? No new roadmap item was visible to this run (JuniorLLM has not moved since the 05:55 note, 09549e1), so this stays on the ask/night/security line.

## Method

Same corpus generator, 0.5 mean|x| absmean ternarization and pair-code encoder as the 10/26 to 10/29 runs (D = 384, U = 50,000 unique rows tiled). N = 2,000,000 docs written as int8 (384 B/row) and pair codes (78 B/row), stored leaves at "last attest", then a 1% append (20,000 docs) gives 2,020,000 rows, 987 chunks of 2,048 rows, 11 dirty chunks, pair 157,560,000 B, int8 775,680,000 B.

Variants, all rebuilding the Merkle root over the pair leaves and decode-comparing the 11 dirty chunks:

- **all_pair+int8_leaves (05:55)**: sha256 every pair and every int8 chunk, compare clean ones with stored leaves (62,464 B of stored leaves).
- **rotate k**: sha256 every pair chunk and compare clean ones with stored pair leaves (31,232 B stored); read int8 only for dirty chunks plus clean chunks with `c % k == night % k`, and decode-compare those against the pair copy. k = 1 (full decode-compare every night), 4, 8, 16.
- **pair leaves only**: no int8 check at all (lower bound, misses every int8-only change).

Five repeats of warm (one untimed run first) and cold (`posix_fadvise(DONTNEED)` on both files) nightly re-attests per variant, night = repeat index so the slice rotates; every timed run returned "agree". Detection: four tamper cases (one trit flipped in the pair copy only, in the int8 copy only, in both consistently, and one int8 byte set from 1 to 5) on three random clean chunks each, starting at a random night, running real nightly re-attests until the change is caught (cap k nights; 2 for pair-only). Bytes restored and all variants re-checked after each case. numpy 2.5.4, 8-vCPU shared box, virtio disk, MemAvailable about 4.4 GB, python3 and numpy only, no network, nothing bound. Index files deleted after the run. A 100,000-doc, 2-repeat smoke run (`python3 bench.py 100000 1 2`) came first. Command, in a scratch folder (not committed):

```
python3 bench.py 2000000 202610291055 5     # N, seed, repeats
```

Full script:

```python
# juniorctl security/night: rotating 1/k decode-compare of clean chunks (no stored int8 leaves)
# vs the 02:55/05:55 full-detection all_pair+int8_leaves re-attest, after a 1% append (python3 + numpy only)
import os, sys, time, hashlib, numpy as np
D=384; U=50000; C=2048; PW=39
P3=3**np.arange(5)
B5=np.array([[(x//3**i)%3 for i in range(5)] for x in range(243)])
V5i=np.where(B5==1,1,np.where(B5==2,-1,0)).astype(np.int8)
V10=np.concatenate([np.repeat(V5i,243,0),np.tile(V5i,(243,1))],1)
def enc_pair(T):
    code=np.where(T==1,1,np.where(T==-1,2,0)).astype(np.uint16); n=len(T)
    b=(np.concatenate([code,np.zeros((n,1),np.uint16)],1).reshape(n,77,5)*P3).sum(2)
    bp=np.concatenate([b,np.zeros((n,1),np.uint16)],1); return (bp[:,0::2]*243+bp[:,1::2]).astype(np.uint16)
dec_pair=lambda r:V10[r].reshape(len(r),390)[:,:D]
RB={"pair":PW*2,"int8":D}; FP={"pair":"idx_pair.bin","int8":"idx_int8.bin"}
def tern(rng,n,Cc):
    X=Cc[rng.integers(0,256,n)]+0.9*rng.standard_normal((n,D)).astype(np.float32); X/=np.linalg.norm(X,axis=1,keepdims=True)
    a=np.abs(X); return np.where(a>0.5*a.mean(1,keepdims=True),np.sign(X),0).astype(np.int8)
def write(T,P,N,mode):
    for k,u in (("int8",T),("pair",P)):
        with open(FP[k],mode) as f:
            for s in range(0,N,len(u)): u[:min(len(u),N-s)].tofile(f)
            f.flush(); os.fsync(f.fileno())
class Idx:
    def __init__(s): s.fd={k:os.open(FP[k],os.O_RDWR) for k in FP}
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
def base(ix,st,dirty,night):          # 05:55 winner: sha256 every pair + int8 leaf, decode-compare dirty only
    ok=True; L=[]
    for c in range(ix.nch()):
        pb=ix.raw("pair",c); ib=ix.raw("int8",c); l=sha(pb); L.append(l)
        if c in dirty: ok&=cmp(pb,ib)
        else: ok&=(l==st["pl"][c]) and (sha(ib)==st["il"][c])
    root(L); return ok
def rot(k):                           # sha256 pair leaves only; decode-compare dirty + clean chunks with c % k == night % k
    def f(ix,st,dirty,night):
        ok=True; L=[]
        for c in range(ix.nch()):
            pb=ix.raw("pair",c); l=sha(pb); L.append(l)
            if c in dirty or (k and c%k==night%k): ok&=cmp(pb,ix.raw("int8",c))
            if c not in dirty: ok&=(l==st["pl"][c])
        root(L); return ok
    return f
V={"all_pair+int8_leaves (05:55)":base,"rotate k=1 (full decode-compare)":rot(1),"rotate k=4":rot(4),"rotate k=8":rot(8),"rotate k=16":rot(16),"pair leaves only (no int8 check)":rot(0)}
def evict():
    for p in FP.values():
        fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
if __name__=="__main__":
    N=int(sys.argv[1]); seed=int(sys.argv[2]); R=int(sys.argv[3]); APP=N//100
    rng=np.random.default_rng(seed); Cc=rng.standard_normal((256,D)).astype(np.float32)
    T=tern(rng,U,Cc); P=enc_pair(T); assert np.array_equal(dec_pair(P),T)
    write(T,P,N,'wb'); ix=Idx()
    st={"pl":{c:sha(ix.raw("pair",c)) for c in range(ix.nch())},"il":{c:sha(ix.raw("int8",c)) for c in range(ix.nch())}}
    print(f"numpy {np.__version__} N {N:,} seed {seed} repeats {R}; MemAvailable {open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]} kB; nproc {os.cpu_count()}",flush=True)
    Ta=tern(rng,APP,Cc); write(Ta,enc_pair(Ta),APP,'ab'); fd0=N//C; dirty=set(range(fd0,ix.nch()))
    print(f"rows {ix.rows():,} chunks {ix.nch()} dirty {len(dirty)} pair {os.path.getsize(FP['pair']):,} B int8 {os.path.getsize(FP['int8']):,} B; stored leaves base {32*2*fd0:,} B rotate {32*fd0:,} B",flush=True)
    res={}
    for r in range(R):
        for mode in ("warm","cold"):
            for n,f in V.items():
                night=r
                if mode=="cold": evict()
                else: f(ix,st,dirty,night)
                t=time.perf_counter(); ok=f(ix,st,dirty,night); dt=time.perf_counter()-t
                assert ok,(mode,n); res.setdefault((mode,n),[]).append(dt)
    print("mode | variant | median ms (spread) | base/variant | int8 bytes read per night")
    nb=lambda n:{"all_pair+int8_leaves (05:55)":ix.nch(),"rotate k=1 (full decode-compare)":ix.nch(),"rotate k=4":len(dirty)+-(-fd0//4),"rotate k=8":len(dirty)+-(-fd0//8),"rotate k=16":len(dirty)+-(-fd0//16),"pair leaves only (no int8 check)":len(dirty)}[n]*C*D
    for mode in ("warm","cold"):
        b=np.median(res[(mode,"all_pair+int8_leaves (05:55)")])
        for n in V: v=res[(mode,n)]; print(f"{mode} | {n} | {np.median(v)*1e3:.1f} ({np.ptp(v)*1e3:.1f}) | {b/np.median(v):.2f}x | {nb(n)/1e6:.0f} MB",flush=True)
    # detection: real tamper on a clean chunk, run nightly re-attest until caught (cap 2k nights)
    trng=np.random.default_rng(seed+1); cases=["pair clean","int8 clean","both clean (consistent)","int8 out-of-alphabet (1 -> 5)"]
    print("tamper | variant | nights to detect (night 0 = tonight), 3 random clean chunks")
    for case in cases:
        for trial in range(3):
            c=int(trng.integers(fd0)); rr=c*C+int(trng.integers(C)); col=int(trng.integers(D)); olds=[]
            row=dec_pair(np.frombuffer(os.pread(ix.fd["pair"],RB["pair"],rr*RB["pair"]),np.uint16).reshape(1,PW)).copy()
            if case.startswith("int8 out"):
                row=np.frombuffer(os.pread(ix.fd["int8"],D,rr*D),np.int8).copy(); row[int(np.flatnonzero(row==1)[0])]=5
                olds.append(("int8",os.pread(ix.fd["int8"],D,rr*D))); os.pwrite(ix.fd["int8"],row.tobytes(),rr*D)
            else:
                row[0,col]=-1 if row[0,col]==1 else 1
                for k in {"pair clean":["pair"],"int8 clean":["int8"],"both clean (consistent)":["pair","int8"]}[case]:
                    olds.append((k,os.pread(ix.fd[k],RB[k],rr*RB[k]))); os.pwrite(ix.fd[k],(enc_pair(row) if k=="pair" else row).tobytes(),rr*RB[k])
            start=int(trng.integers(16)); out=[]
            for n,f in V.items():
                kk={"rotate k=4":4,"rotate k=8":8,"rotate k=16":16}.get(n,1); got="never"
                for d in range(kk if n!="pair leaves only (no int8 check)" else 2):
                    if not f(ix,st,dirty,start+d): got=str(d); break
                out.append(f"{n}: {got}")
            print(f"{case} | chunk {c} | "+"; ".join(out),flush=True)
            for k,o in olds: os.pwrite(ix.fd[k],o,rr*RB[k])
            assert all(f(ix,st,dirty,start) for f in V.values())
    ix.close(); [os.remove(p) for p in FP.values()]
```

## Results

Nightly re-attest, 2,020,000 rows, 1% append, 5 repeats:

| mode | variant | median ms (spread) | baseline / variant | int8 read per night |
|---|---|---|---|---|
| warm | all_pair+int8_leaves (05:55) | 781.6 (250.1) | 1.00x | 776 MB |
| warm | rotate k=1 (full decode-compare) | 1,262.2 (128.9) | 0.62x | 776 MB |
| warm | rotate k=4 | 406.4 (123.0) | 1.92x | 201 MB |
| warm | **rotate k=8** | **273.0 (18.8)** | **2.86x** | 105 MB |
| warm | rotate k=16 | 203.3 (37.8) | 3.85x | 57 MB |
| warm | pair leaves only (no int8 check) | 139.1 (9.1) | 5.62x | 9 MB |
| cold | all_pair+int8_leaves (05:55) | 869.8 (350.8) | 1.00x | 776 MB |
| cold | rotate k=1 (full decode-compare) | 1,430.5 (191.4) | 0.61x | 776 MB |
| cold | rotate k=4 | 679.5 (212.3) | 1.28x | 201 MB |
| cold | **rotate k=8** | **418.1 (131.4)** | **2.08x** | 105 MB |
| cold | rotate k=16 | 296.0 (87.5) | 2.94x | 57 MB |
| cold | pair leaves only (no int8 check) | 172.4 (19.4) | 5.05x | 9 MB |

Nights to detect (0 = the night of the change), three random clean chunks per case:

| case | 05:55 baseline | k=1 | k=4 | k=8 | k=16 | pair only |
|---|---|---|---|---|---|---|
| pair only, 1 trit | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| int8 only, 1 trit | 0, 0, 0 | 0, 0, 0 | 2, 0, 3 | 6, 0, 3 | 14, 8, 3 | never |
| both, consistent 1 trit | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| int8 byte 1 to 5 (out of alphabet) | 0, 0, 0 | 0, 0, 0 | 2, 2, 0 | 6, 2, 4 | 6, 10, 4 | never |

Worst-case latency for an int8-only change is k - 1 nights by construction (mean (k - 1) / 2); every observed value is inside that bound. Any change that touches the pair copy is still caught the same night by every variant.

## Verdict

**Viable, at k = 8.** Rotating a 1/8 decode-compare slice of the int8 copy against the sha256-verified pair-code leaves is 2.86x faster warm (273 ms vs 782 ms) and 2.08x faster cold (418 ms vs 870 ms) than `all_pair+int8_leaves`, reads 105 MB of int8 per night instead of 776 MB, halves the stored leaves (31 KB vs 62 KB), and still catches every pair-side or consistent tamper the same night. The cost is a stated bound: an int8-only change (including an out-of-alphabet byte) is caught within 7 nights instead of the same night. k = 4 clears 1.5x warm (1.92x) but not cold (1.28x); k = 16 is 2.9 to 3.9x but stretches the bound to 15 nights. Full decode-compare every night (k = 1) is 0.6x, so the decode is only worth paying on a slice. This works because the ternary pair copy is small enough (5x smaller than int8) to hash in full every night and serve as the trust anchor, so the big int8 copy only needs spot coverage.

Caveats: single shared 8-vCPU box with noisy spreads (up to 350 ms on the cold baseline). The int8 copy is what `night` actually sweeps, so a 7-night window for an int8-only change is a policy call for `security`, not a free lunch; a same-night alarm could still come from a cheap full int8 check on demand (for example after any write outside the append path). Three tamper positions per case is a sanity check, not a latency distribution; the bound itself is deterministic.

## Pair files this wave

- No pair file was visible to this run for the 10:05 wave. JuniorLLM has not moved since the 05:55 note ([09549e1](https://github.com/cloudcover95/JuniorLLM/commit/09549e1)), and this note adds one new file under `junior_osai/vault/findings/` only. The ask/security side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

With k = 8 as the nightly default, can the slice be chosen smarter than `c % k`: weight it toward chunks the `night` sweep actually ranks into its top results (or chunks written longest ago), and does that cut the expected nights-to-detect for changes that would matter to `ask` answers by at least 1.5x at the same 105 MB/night int8 read budget?

close: 4ed4d4209e2c96e3 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-29_0555)
