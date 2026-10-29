---
local: true
date: 2026-10-29
wave: "2026-10-29 02:05 (bench 02:55, ran 02:58 to 03:00)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`security` re-attesting the packed ask index that `night` sweeps as int8)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, merkle, sha256, juniorctl, security, night, attestation, tamper, incremental, bench]
---

# juniorctl security: chunked Merkle re-attest of pair codes + dirty-chunk decode-compare vs full rehash + full cross-check

## Question

The 18:55 note ([[2026-10-28_1855_security-attest-ask-index-pair-codes-vs-i2s-vs-int8]]) asked this next: if `security` attests the packed (pair-code) index and `night` sweeps the int8 copy, the two must agree. Is a chunked (2,048-row) Merkle digest over pair codes, with per-chunk decode-and-compare against int8 only for chunks whose leaf changed, at least 1.5x cheaper per nightly re-attest than a full packed sha256 plus a full decode cross-check, after a 1% daily append? The 18:55 note left out the cost of that cross-check (one full decode sweep), so this run measures it. A re-attest that is cheaper only because it stops looking at untouched chunks would miss tampering there, so tamper detection is measured for each variant and any variant that loses detection the baseline has is not counted as viable. The 02:05 Chief run gave both pairs "no gap" (25th idle wave, roadmap blobs unchanged at 4efb666 / bac6923), so this stays on the ask/night/security line.

## Method

Same corpus generator, 0.5 mean|x| absmean ternarization and pair-code encoder as the 10/26 to 10/28 runs (D = 384, U = 50,000 unique rows tiled). N = 2,000,000 docs are written as int8 (384 B/row) and pair codes (39 uint16 = 78 B/row). "Last attest" state is then stored: a sha256 leaf for every 2,048-row chunk of both copies. A 1% daily append (20,000 new docs, new draws) goes onto both files, so 2,020,000 rows, 987 chunks, pair 157,560,000 B, int8 775,680,000 B. The dirty set is 11 chunks: the old partial last chunk plus 10 new ones. A second scenario adds 16 legitimate in-place edits: a writer rewrites one row in each of 16 random old chunks in both copies and journals those chunks as dirty (27 dirty chunks).

Four re-attest variants, all reading 2,048-row chunks with `os.pread`. Leaves are sha256, and the root is a binary sha256 tree over the leaves with a 0x01 prefix:

- **full** (baseline): one pass that streams a sha256 over the whole pair file and decodes every pair row (59049x10 table) and compares it bit-exact to the int8 chunk.
- **dirty_only**: rehash only the dirty pair leaves, recompute the root, decode-compare only dirty chunks.
- **all_pair_leaves**: rehash every pair leaf (each clean leaf must equal its stored value), recompute the root, decode-compare only dirty chunks.
- **all_pair+int8_leaves**: as all_pair_leaves, but also rehash every int8 leaf (each clean int8 leaf must equal its stored value).

Warm = one untimed run of the same variant first (page cache). Cold = `posix_fadvise(DONTNEED)` on both files right before timing. Five repeats per scenario × mode × variant; every timed run returned "agree". Tamper tests run in the append+edit state. One trit is flipped and re-encoded in place in: a clean old chunk of the pair copy only; a clean old chunk of the int8 copy only; a clean old chunk of both copies consistently (same flip in each); and a dirty (edited) chunk of the pair copy. Each variant is run, the bytes are restored, and the run checks that every variant passes again.

numpy 2.5.4, 8-vCPU Xeon box with virtio disk, MemAvailable about 2.47 GB at start (enough for N = 2M because everything streams per chunk, so no scale-down was needed), python3 and numpy only, no network, nothing bound, hashlib single-threaded. Index files deleted after the run. Command, in a scratch folder (not committed); a 100,000-doc, 2-repeat smoke run (`python3 bench.py 100000 1 2 4`) came first:

```
python3 bench.py 2000000 202610290255 5 16     # N, seed, repeats, in-place edits
```

Full script:

```python
# juniorctl security/night: chunked (2,048-row) Merkle re-attest over pair codes + dirty-chunk decode-compare vs int8,
# against a full packed sha256 + full decode cross-check, after a 1% append (python3 + numpy only)
import os, sys, time, hashlib, numpy as np
D=384; U=50000; C=2048; PW=39  # pair row = 39 uint16 = 78 B; int8 row = 384 B
P3=3**np.arange(5)
B5=np.array([[(x//3**i)%3 for i in range(5)] for x in range(243)])
V5i=np.where(B5==1,1,np.where(B5==2,-1,0)).astype(np.int8)
V10=np.concatenate([np.repeat(V5i,243,0),np.tile(V5i,(243,1))],1)          # 59049 x 10 pair table
def enc_pair(T):
    code=np.where(T==1,1,np.where(T==-1,2,0)).astype(np.uint16); n=len(T)
    b=(np.concatenate([code,np.zeros((n,1),np.uint16)],1).reshape(n,77,5)*P3).sum(2)
    bp=np.concatenate([b,np.zeros((n,1),np.uint16)],1); return (bp[:,0::2]*243+bp[:,1::2]).astype(np.uint16)
dec_pair=lambda r:V10[r].reshape(len(r),390)[:,:D]
RB={"pair":PW*2,"int8":D}
FP={"pair":"idx_pair.bin","int8":"idx_int8.bin"}
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
def leaf(b): return hashlib.sha256(b).digest()
def root(L):
    L=list(L)
    while len(L)>1:
        if len(L)&1: L.append(L[-1])
        L=[hashlib.sha256(b"\x01"+L[i]+L[i+1]).digest() for i in range(0,len(L),2)]
    return L[0]
def cmp(pb,ib):
    p=np.frombuffer(pb,np.uint16).reshape(-1,PW); t=np.frombuffer(ib,np.int8).reshape(-1,D)
    return np.array_equal(dec_pair(p),t)
# ---- the four re-attest variants; each returns (ok, bytes_read) ----
def full(ix,st,dirty):           # baseline: sha256 of whole pair file + decode every row bit-exact vs int8 (one pass)
    h=hashlib.sha256(); ok=True; nb=0
    for c in range(ix.nch()):
        pb=ix.raw("pair",c); ib=ix.raw("int8",c); h.update(pb); nb+=len(pb)+len(ib); ok&=cmp(pb,ib)
    st["file"]=h.digest(); return ok,nb
def dirty_only(ix,st,dirty):     # rehash only dirty pair leaves, recompute root, decode-compare only dirty chunks
    L=dict(st["pl"]); ok=True; nb=0
    for c in dirty:
        pb=ix.raw("pair",c); ib=ix.raw("int8",c); nb+=len(pb)+len(ib); L[c]=leaf(pb); ok&=cmp(pb,ib)
    st["root"]=root(L[i] for i in range(ix.nch())); return ok,nb
def all_pair(ix,st,dirty):       # rehash ALL pair leaves (clean ones must match stored), decode-compare dirty only
    ok=True; nb=0; L=[]
    for c in range(ix.nch()):
        pb=ix.raw("pair",c); nb+=len(pb); l=leaf(pb); L.append(l)
        if c in dirty: ib=ix.raw("int8",c); nb+=len(ib); ok&=cmp(pb,ib)
        else: ok&=(l==st["pl"][c])
    st["root"]=root(L); return ok,nb
def all_both(ix,st,dirty):       # as all_pair, plus rehash ALL int8 leaves (clean ones must match stored)
    ok=True; nb=0; L=[]
    for c in range(ix.nch()):
        pb=ix.raw("pair",c); ib=ix.raw("int8",c); nb+=len(pb)+len(ib); l=leaf(pb); L.append(l)
        if c in dirty: ok&=cmp(pb,ib)
        else: ok&=(l==st["pl"][c]) and (leaf(ib)==st["il"][c])
    st["root"]=root(L); return ok,nb
VAR={"full":full,"dirty_only":dirty_only,"all_pair_leaves":all_pair,"all_pair+int8_leaves":all_both}
def evict():
    for p in FP.values():
        fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
def flip(ix,k,r,col):            # flip one trit of row r in copy k (re-encode in place)
    pr=np.frombuffer(os.pread(ix.fd["pair"],RB["pair"],r*RB["pair"]),np.uint16).reshape(1,PW); row=dec_pair(pr).copy()
    if k=="int8": row=np.frombuffer(os.pread(ix.fd["int8"],D,r*D),np.int8).reshape(1,D).copy()
    old=(os.pread(ix.fd[k],RB[k],r*RB[k]))
    row[0,col]=-1 if row[0,col]==1 else 1
    os.pwrite(ix.fd[k],(enc_pair(row) if k=="pair" else row).tobytes(),r*RB[k]); return old
if __name__=="__main__":
    N=int(sys.argv[1]); seed=int(sys.argv[2]); R=int(sys.argv[3]); APP=N//100; NED=int(sys.argv[4]) if len(sys.argv)>4 else 16
    rng=np.random.default_rng(seed); Cc=rng.standard_normal((256,D)).astype(np.float32)
    T=tern(rng,U,Cc); P=enc_pair(T); assert np.array_equal(dec_pair(P),T)
    write(T,P,N,'wb'); ix=Idx()
    st={"pl":{c:leaf(ix.raw("pair",c)) for c in range(ix.nch())},"il":{c:leaf(ix.raw("int8",c)) for c in range(ix.nch())}}  # stored at last attest
    n0ch=ix.nch(); print(f"numpy {np.__version__} N {N:,} seed {seed} repeats {R} chunk {C} rows; MemAvailable {open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]} kB",flush=True)
    Ta=tern(rng,APP,Cc); write(Ta,enc_pair(Ta),APP,'ab')                      # 1% daily append to both copies
    first_dirty=N//C; dirty_app=set(range(first_dirty,ix.nch()))
    print(f"after append: rows {ix.rows():,} chunks {ix.nch()} pair {os.path.getsize(FP['pair']):,} B int8 {os.path.getsize(FP['int8']):,} B dirty(append) {len(dirty_app)}",flush=True)
    res={}
    def run(scen,dirty):
        for r in range(R):
            for mode in ("warm","cold"):
                for name,fn in VAR.items():
                    if mode=="cold": evict()
                    else: fn(ix,dict(st),dirty)
                    s=dict(st); t=time.perf_counter(); ok,nb=fn(ix,s,dirty); dt=time.perf_counter()-t
                    assert ok,(scen,name); res.setdefault((scen,mode,name),[]).append((dt,nb))
    run("append 1%",dirty_app)
    # in-place edit: legit writer rewrites one row (both copies, consistently) in NED random old chunks, journals them dirty
    ed=set(int(c) for c in rng.choice(first_dirty,NED,replace=False)); Te=tern(rng,len(ed),Cc)
    for c,row in zip(sorted(ed),Te):
        r=c*C+int(rng.integers(C)); os.pwrite(ix.fd["int8"],row.tobytes(),r*D); os.pwrite(ix.fd["pair"],enc_pair(row[None]).tobytes(),r*RB["pair"])
    dirty_ed=dirty_app|ed; run(f"append 1% + {NED} in-place edits",dirty_ed)
    print("scenario | mode | variant | dirty | median ms (spread) | bytes read | full/variant")
    for scen,dirty in (("append 1%",dirty_app),(f"append 1% + {NED} in-place edits",dirty_ed)):
        for mode in ("warm","cold"):
            fm=np.median([a for a,_ in res[(scen,mode,"full")]])
            for name in VAR:
                v=np.array(res[(scen,mode,name)]); print(f"{scen} | {mode} | {name} | {len(dirty)} | {np.median(v[:,0])*1e3:9.1f} ({np.ptp(v[:,0])*1e3:6.1f}) | {int(v[0,1]):,} | {fm/np.median(v[:,0]):6.2f}x",flush=True)
    # tamper: single-trit flip in a CLEAN old chunk (pair only, int8 only, both consistently) and in a DIRTY chunk (pair only)
    clean=[c for c in range(first_dirty) if c not in dirty_ed]; trng=np.random.default_rng(seed+1); det={}
    for case in ("pair clean","int8 clean","both clean (consistent)","pair dirty"):
        c=int(trng.choice(clean)) if "clean" in case else int(trng.choice(sorted(ed))); r=c*C+int(trng.integers(C)); col=int(trng.integers(D))
        ks=["pair","int8"] if case.startswith("both") else [case.split()[0]]; olds=[(k,flip(ix,k,r,col)) for k in ks]
        for name,fn in VAR.items(): det[(case,name)]=not fn(ix,dict(st),dirty_ed)[0]
        for k,o in olds: os.pwrite(ix.fd[k],o,r*RB[k])
        assert all(fn(ix,dict(st),dirty_ed)[0] for fn in VAR.values())
    print("tamper detected (case | "+" | ".join(VAR)+")")
    for case in ("pair clean","int8 clean","both clean (consistent)","pair dirty"): print(case,"|"," | ".join(str(det[(case,n)]) for n in VAR))
    ix.close(); [os.remove(p) for p in FP.values()]
```

## Results

Medians in ms over 5 repeats (spread, max minus min, in parentheses). Bytes = bytes read per re-attest. Ratio = full median / variant median.

| scenario | mode | variant | dirty chunks | median ms (spread) | bytes read | full / variant |
|---|---|---|---|---|---|---|
| append 1% | warm | full | 11 | 1,174.1 (211.8) | 933,240,000 | 1.00x |
| append 1% | warm | dirty_only | 11 | 12.5 (4.5) | 9,772,224 | 93.72x |
| append 1% | warm | all_pair_leaves | 11 | 128.0 (58.3) | 165,682,368 | 9.17x |
| append 1% | warm | all_pair+int8_leaves | 11 | 689.4 (37.0) | 933,240,000 | **1.70x** |
| append 1% | cold | full | 11 | 1,288.5 (290.3) | 933,240,000 | 1.00x |
| append 1% | cold | dirty_only | 11 | 18.1 (6.1) | 9,772,224 | 71.27x |
| append 1% | cold | all_pair_leaves | 11 | 167.6 (20.9) | 165,682,368 | 7.69x |
| append 1% | cold | all_pair+int8_leaves | 11 | 789.4 (28.3) | 933,240,000 | **1.63x** |
| append 1% + 16 edits | warm | full | 27 | 1,158.8 (92.2) | 933,240,000 | 1.00x |
| append 1% + 16 edits | warm | dirty_only | 27 | 31.0 (6.3) | 24,911,040 | 37.44x |
| append 1% + 16 edits | warm | all_pair_leaves | 27 | 143.5 (15.1) | 178,265,280 | 8.07x |
| append 1% + 16 edits | warm | all_pair+int8_leaves | 27 | 693.1 (46.4) | 933,240,000 | **1.67x** |
| append 1% + 16 edits | cold | full | 27 | 1,320.8 (290.1) | 933,240,000 | 1.00x |
| append 1% + 16 edits | cold | dirty_only | 27 | 58.2 (12.0) | 24,911,040 | 22.71x |
| append 1% + 16 edits | cold | all_pair_leaves | 27 | 199.7 (49.6) | 178,265,280 | 6.61x |
| append 1% + 16 edits | cold | all_pair+int8_leaves | 27 | 856.4 (133.1) | 933,240,000 | **1.54x** |

Single-trit tamper detection (True = re-attest fails):

| tamper | full | dirty_only | all_pair_leaves | all_pair+int8_leaves |
|---|---|---|---|---|
| pair copy, clean old chunk | True | **False** | True | True |
| int8 copy, clean old chunk | True | **False** | **False** | True |
| both copies consistently, clean old chunk | **False** | **False** | True | True |
| pair copy, dirty chunk | True | True | True | True |

A 100,000-doc smoke run gave the same ordering and the same detection table (all_pair+int8_leaves at 1.32x to 1.72x, full at 53 to 68 ms).

## Verdict

**Viable, but only for the full-detection variant and with a thin margin.** The cheapest candidate is not acceptable. dirty_only is 22x to 94x faster, but it misses every tamper outside the dirty chunks (pair or int8), so its speed comes from not looking. all_pair_leaves (Merkle over the packed copy, decode-compare only dirty chunks) is 6.6x to 9.2x faster and reads 5.2x to 5.6x fewer bytes. It catches pair tamper anywhere, and also a consistent two-copy tamper that the baseline cannot see (the baseline only checks that the two copies agree, so it has no stored reference). It misses an int8-only flip in a clean chunk, which is exactly the "`night` sweeps a copy that no longer matches the attested one" failure this question is about. all_pair+int8_leaves catches all four tampers, a strict superset of the baseline's detection. It is **1.54x to 1.70x** faster (warm 1.67x to 1.70x, cold 1.54x to 1.63x) at the same 933 MB read, so the whole gain comes from replacing the full pair→int8 decode-compare (about 1,050 ms warm at 2M rows) with an int8 sha256 (about 560 ms) plus decoding only 11 to 27 chunks, a saving of about 485 ms warm. It clears 1.5x in every cell, so the verdict is viable. Practical rule for `juniorctl security`: store per-chunk sha256 leaves for both copies at attest time, rehash all leaves nightly, and decode-compare only the journaled dirty chunks. Never skip clean leaves.

Caveats: the worst cell (cold, with edits) is 1.54x, with a full-baseline spread of 290 ms and a candidate spread of 133 ms on a shared, noisy box, so a heavier day or a busy neighbour could push it under 1.5x. The in-place edit case relies on the writer journaling dirty chunks. An unjournaled legitimate edit would show up as a tamper (fail-safe, not silent). The virtio disk is probably host-cached, so the cold reads are optimistic. Both sides are byte-equal here, so slower real disks would shrink the ratio rather than widen it. Data is synthetic tiled docs. The 1.70x/9x split shows the int8 rehash dominates the candidate's cost (≈560 ms of ≈690 ms warm).

## Pair files this wave

- The 02:05 Chief gave both pairs "no gap" (25th idle wave), so this wave has no pair file to link. JuniorLLM has not moved since the 18:55 note ([2cd0c88](https://github.com/cloudcover95/JuniorLLM/commit/2cd0c88)). The only other push this wave was JuniorHome ca97f84 (LLC, scripts/advance_1900_second_slice_receipt_45min_named.py), whose path this note does not touch. The ask/security side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

The int8 leaf rehash is now most of the full-detection re-attest. Can the int8 side be checked more cheaply while keeping every clean-chunk int8 flip detectable? Candidate: store per-chunk leaves as sha256 of the int8 chunk re-encoded to pair codes on the fly (78 B/row hashed instead of 384 B/row), compared with a blake2b-256 int8 leaf and with the current sha256 int8 leaf. Does it bring all_pair+int8_leaves to at least 1.5x over itself at 2M rows, warm and cold, with the same four-tamper detection table?

close: 3df9e6388c9e6e2f (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-28_1855)
