---
local: true
date: 2026-10-29
wave: "2026-10-29 05:05 (bench 05:55, ran 05:56 to 06:05)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`security` re-attesting the packed ask index that `night` sweeps as int8)"
verdict: not-viable
tags: [osai, vault, findings, ternary, pair-codes, i2s, int8, merkle, sha256, blake2b, juniorctl, security, night, attestation, tamper, bench]
---

# juniorctl security: cheaper int8 leaf for the full-detection Merkle re-attest (blake2b, pair re-encode, 2-bit pack) vs plain sha256

## Question

The 02:55 note ([[2026-10-29_0255_security-merkle-reattest-pair-leaves-dirty-decode-vs-full-crosscheck]]) found that rehashing every int8 leaf is most of the cost of the only re-attest variant that keeps full tamper detection (`all_pair+int8_leaves`, about 560 ms of about 690 ms warm). Its next question: can the int8 leaf be made cheaper by hashing a ternary-packed form of the int8 chunk (pair codes at 78 B/row, or I2_S-style 2-bit at 96 B/row) instead of the raw 384 B/row, or by switching the hash to blake2b-256, and does any of these bring `all_pair+int8_leaves` to at least 1.5x faster than itself at 2M rows, warm and cold, without losing any tamper case? No new roadmap item was visible to this run (the quant team has been idle since 10-24 with roadmap blobs unchanged at 4efb666 / bac6923), so this stays on the ask/night/security line.

## Method

Same corpus generator, 0.5 mean|x| absmean ternarization and pair-code encoder as the 10/26 to 10/29 runs (D = 384, U = 50,000 unique rows tiled). N = 2,000,000 docs written as int8 (384 B/row) and pair codes (78 B/row), stored leaves computed at "last attest", then a 1% append (20,000 docs) gives 2,020,000 rows, 987 chunks of 2,048 rows, 11 dirty chunks, pair 157,560,000 B, int8 775,680,000 B.

Every variant is the 02:55 `all_pair+int8_leaves` re-attest (sha256 every pair leaf, compare clean leaves with stored ones, decode-compare dirty chunks against int8, rebuild the Merkle root), and only the int8 leaf function changes:

- **sha256 int8** (02:55 baseline): sha256 of the raw 786,432 B chunk.
- **blake2b-256 int8**: hashlib blake2b, digest 32 B, same bytes.
- **sha256 of pair re-encode**: int8 chunk re-encoded to pair codes with numpy, then sha256 of the 78 B/row result.
- **sha256 of 2-bit pack**: `(u8 & 3)` gives {0, 1, 3} for {0, 1, -1}, packed 4 per byte with shifts and `bitwise_or.reduce` (96 B/row), then sha256.
- **sha256 of 2-bit pack + alphabet check**: as above, but a chunk containing any byte outside {0, 1, 255} gets a poison leaf.

A leaf-only microbenchmark first times each function over 64 warm in-memory int8 chunks (50.3 MB, no IO). Then five repeats of warm (one untimed run first) and cold (`posix_fadvise(DONTNEED)` on both files) full re-attests per variant; every timed run returned "agree". Tamper cases on a clean old chunk: one trit flipped in the pair copy only, in the int8 copy only, in both consistently, and one int8 byte set from 1 to 5 (out of the trit alphabet). Bytes are restored and all variants re-checked after each case. numpy 2.5.4, 8-vCPU Xeon box, virtio disk, MemAvailable about 6.4 GB, python3 and numpy only, no network, nothing bound, hashlib single-threaded. Index files deleted after the run. A 100,000-doc, 2-repeat smoke run (`python3 bench.py 100000 1 2`) came first. Command, in a scratch folder (not committed):

```
python3 bench.py 2000000 202610290555 5     # N, seed, repeats
```

Full script:

```python
# juniorctl security/night: cheaper int8 leaf for the full-detection chunked Merkle re-attest.
# all_pair+int8_leaves (02:55 winner) with the int8 leaf computed 5 ways, after a 1% append (python3 + numpy only)
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
SH=np.array([0,2,4,6],np.uint8)
def pack2(ib):                       # I2_S-style 2-bit pack of int8 trits: (u8 & 3) -> {0,1,3}, 4 per byte (96 B/row)
    a=np.frombuffer(ib,np.uint8).reshape(-1,4)&3; return np.bitwise_or.reduce(a<<SH,axis=1).tobytes()
def pack2_chk(ib):                   # same, plus alphabet check (any byte not in {0,1,255} -> poison leaf)
    u=np.frombuffer(ib,np.uint8); bad=bool(np.count_nonzero((u>1)&(u<255)))
    return b"BAD" if bad else pack2(ib)
IL={"sha256 int8 (02:55)":lambda b:sha(b),
    "blake2b-256 int8":lambda b:hashlib.blake2b(b,digest_size=32).digest(),
    "sha256 of pair re-encode":lambda b:sha(enc_pair(np.frombuffer(b,np.int8).reshape(-1,D)).tobytes()),
    "sha256 of 2-bit pack":lambda b:sha(pack2(b)),
    "sha256 of 2-bit pack + alphabet check":lambda b:sha(pack2_chk(b))}
def cmp(pb,ib): return np.array_equal(dec_pair(np.frombuffer(pb,np.uint16).reshape(-1,PW)),np.frombuffer(ib,np.int8).reshape(-1,D))
def root(L):
    L=list(L)
    while len(L)>1:
        if len(L)&1: L.append(L[-1])
        L=[hashlib.sha256(b"\x01"+L[i]+L[i+1]).digest() for i in range(0,len(L),2)]
    return L[0]
def reattest(ix,st,dirty,name):
    f=IL[name]; ok=True; L=[]; il=st["il"][name]
    for c in range(ix.nch()):
        pb=ix.raw("pair",c); ib=ix.raw("int8",c); l=sha(pb); L.append(l)
        if c in dirty: ok&=cmp(pb,ib)
        else: ok&=(l==st["pl"][c]) and (f(ib)==il[c])
    root(L); return ok
def evict():
    for p in FP.values():
        fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
if __name__=="__main__":
    N=int(sys.argv[1]); seed=int(sys.argv[2]); R=int(sys.argv[3]); APP=N//100
    rng=np.random.default_rng(seed); Cc=rng.standard_normal((256,D)).astype(np.float32)
    T=tern(rng,U,Cc); P=enc_pair(T); assert np.array_equal(dec_pair(P),T)
    write(T,P,N,'wb'); ix=Idx()
    for b in [ix.raw("int8",c) for c in range(3)]:   # pair re-encode / 2-bit pack must be injective on trit data
        t=np.frombuffer(b,np.int8).reshape(-1,D); assert cmp(enc_pair(t).tobytes(),b)
    st={"pl":{c:sha(ix.raw("pair",c)) for c in range(ix.nch())},
        "il":{n:{c:f(ix.raw("int8",c)) for c in range(ix.nch())} for n,f in IL.items()}}
    print(f"numpy {np.__version__} N {N:,} seed {seed} repeats {R}; MemAvailable {open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]} kB",flush=True)
    Ta=tern(rng,APP,Cc); write(Ta,enc_pair(Ta),APP,'ab'); fd0=N//C; dirty=set(range(fd0,ix.nch()))
    print(f"rows {ix.rows():,} chunks {ix.nch()} dirty {len(dirty)} pair {os.path.getsize(FP['pair']):,} B int8 {os.path.getsize(FP['int8']):,} B",flush=True)
    # leaf-only microbench: int8 leaf cost over a warm 64-chunk sample (no IO)
    smp=[ix.raw("int8",c) for c in range(min(64,fd0))]; leafr={}
    for n,f in IL.items():
        ts=[]
        for r in range(R):
            t=time.perf_counter(); [f(b) for b in smp]; ts.append(time.perf_counter()-t)
        leafr[n]=ts
    print("int8 leaf only (sample chunks) | median ms (spread) | MB/s | hashed B/row")
    hb={"sha256 int8 (02:55)":384,"blake2b-256 int8":384,"sha256 of pair re-encode":78,"sha256 of 2-bit pack":96,"sha256 of 2-bit pack + alphabet check":96}
    for n,ts in leafr.items(): m=np.median(ts); print(f"{n} | {m*1e3:.1f} ({np.ptp(ts)*1e3:.1f}) | {len(smp)*C*D/m/1e6:.0f} | {hb[n]}",flush=True)
    res={}
    for r in range(R):
        for mode in ("warm","cold"):
            for n in IL:
                if mode=="cold": evict()
                else: reattest(ix,st,dirty,n)
                t=time.perf_counter(); ok=reattest(ix,st,dirty,n); dt=time.perf_counter()-t
                assert ok,(mode,n); res.setdefault((mode,n),[]).append(dt)
    print("mode | int8 leaf | median ms (spread) | base/variant")
    for mode in ("warm","cold"):
        b=np.median(res[(mode,"sha256 int8 (02:55)")])
        for n in IL: v=res[(mode,n)]; print(f"{mode} | {n} | {np.median(v)*1e3:.1f} ({np.ptp(v)*1e3:.1f}) | {b/np.median(v):.2f}x",flush=True)
    clean=list(range(fd0)); trng=np.random.default_rng(seed+1); det={}
    def tamper(case):
        c=int(trng.choice(clean)); r=c*C+int(trng.integers(C)); col=int(trng.integers(D)); olds=[]
        row=dec_pair(np.frombuffer(os.pread(ix.fd["pair"],RB["pair"],r*RB["pair"]),np.uint16).reshape(1,PW)).copy()
        if case=="int8 out-of-alphabet (1 -> 5)":
            row=np.frombuffer(os.pread(ix.fd["int8"],D,r*D),np.int8).copy(); cols=np.flatnonzero(row==1); col=int(cols[0]); row[col]=5
            olds.append(("int8",os.pread(ix.fd["int8"],D,r*D))); os.pwrite(ix.fd["int8"],row.tobytes(),r*D); return r,olds
        row[0,col]=-1 if row[0,col]==1 else 1
        ks={"pair clean":["pair"],"int8 clean":["int8"],"both clean (consistent)":["pair","int8"]}[case]
        for k in ks:
            olds.append((k,os.pread(ix.fd[k],RB[k],r*RB[k]))); os.pwrite(ix.fd[k],(enc_pair(row) if k=="pair" else row).tobytes(),r*RB[k])
        return r,olds
    cases=["pair clean","int8 clean","both clean (consistent)","int8 out-of-alphabet (1 -> 5)"]
    for case in cases:
        r,olds=tamper(case)
        for n in IL: det[(case,n)]=not reattest(ix,st,dirty,n)
        for k,o in olds: os.pwrite(ix.fd[k],o,r*RB[k])
        assert all(reattest(ix,st,dirty,n) for n in IL)
    print("tamper detected | "+" | ".join(IL))
    for case in cases: print(case,"|"," | ".join(str(det[(case,n)]) for n in IL))
    ix.close(); [os.remove(p) for p in FP.values()]
```

## Results

int8 leaf function alone (64 warm chunks, 50.3 MB, 5 repeats):

| int8 leaf | median ms (spread) | MB/s of int8 input | hashed B/row |
|---|---|---|---|
| sha256 int8 (02:55) | 28.6 (1.3) | 1,763 | 384 |
| blake2b-256 int8 | 64.9 (3.5) | 775 | 384 |
| sha256 of pair re-encode | 584.3 (16.5) | 86 | 78 |
| sha256 of 2-bit pack | 234.5 (2.5) | 215 | 96 |
| sha256 of 2-bit pack + alphabet check | 248.6 (8.7) | 202 | 96 |

Full `all_pair+int8_leaves` re-attest, 2,020,000 rows, 1% append, 5 repeats:

| mode | int8 leaf | median ms (spread) | baseline / variant |
|---|---|---|---|
| warm | sha256 int8 (02:55) | 700.1 (47.4) | 1.00x |
| warm | blake2b-256 int8 | 1,314.2 (123.3) | 0.53x |
| warm | sha256 of pair re-encode | 9,619.6 (607.4) | 0.07x |
| warm | sha256 of 2-bit pack | 4,075.4 (563.1) | 0.17x |
| warm | sha256 of 2-bit pack + alphabet check | 4,168.0 (740.7) | 0.17x |
| cold | sha256 int8 (02:55) | 891.9 (173.7) | 1.00x |
| cold | blake2b-256 int8 | 1,444.2 (155.4) | 0.62x |
| cold | sha256 of pair re-encode | 9,413.1 (1,615.9) | 0.09x |
| cold | sha256 of 2-bit pack | 4,111.2 (698.6) | 0.22x |
| cold | sha256 of 2-bit pack + alphabet check | 4,238.8 (457.3) | 0.21x |

Tamper detected (clean old chunk):

| case | sha256 int8 | blake2b-256 | pair re-encode | 2-bit pack | 2-bit pack + check |
|---|---|---|---|---|---|
| pair only, 1 trit | yes | yes | yes | yes | yes |
| int8 only, 1 trit | yes | yes | yes | yes | yes |
| both, consistent 1 trit | yes | yes | yes | yes | yes |
| int8 byte 1 to 5 (out of alphabet) | yes | yes | yes | **no** | yes |

## Verdict

**Not viable.** Hashing fewer bytes does not pay here: sha256 on this box runs at about 1.76 GB/s (hardware SHA extensions), so the raw 384 B/row int8 leaf is already cheaper than any numpy repack. Re-encoding to pair codes is about 20x slower per leaf and makes the re-attest 0.07 to 0.09x of the baseline. The 2-bit pack is about 8x slower per leaf (0.17 to 0.22x end to end). Without its alphabet check, it silently misses an out-of-alphabet int8 byte (1 to 5 aliases to 1 under `& 3`), so the check is required, and that makes it slightly slower again. blake2b-256 is about half the speed of sha256 on this CPU (0.53 to 0.62x). Keep plain sha256 over the raw int8 chunk as the int8 leaf. The ternary win stays where the 10/28 and 10/29 notes put it: storing and attesting the pair-code copy (4.8 to 5.1x vs int8), not repacking int8 at attest time.

Caveats: single shared 8-vCPU box with noisy spreads (up to 1.6 s on the cold pair re-encode). The numpy repacks are pure-Python-level vectorized code with temporaries, and a C or SIMD packer could be much faster, but this lane is python3 and numpy only. The sha256 vs blake2b ordering depends on the CPU having SHA extensions, and on a CPU without them blake2b could win.

## Pair files this wave

- No pair file was visible to this run for the 05:05 wave. JuniorLLM has not moved since the 02:55 note ([ffed622](https://github.com/cloudcover95/JuniorLLM/commit/ffed622)), and this note adds one new file under `junior_osai/vault/findings/` only. The ask/security side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Since the int8 leaf can't get cheaper, can it be checked less often without losing detection? Candidate: drop the separate int8 leaves and instead decode-compare a rotating 1/k slice of clean chunks each night against the already-verified pair leaves (full int8 coverage every k nights), and measure nightly cost vs `all_pair+int8_leaves` at k = 4 and 8, plus the expected time to catch a single clean-chunk int8 flip. Is any k at least 1.5x cheaper per night with a stated detection-latency bound?

close: 6b4400dbafc526c9 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-29_0255)
