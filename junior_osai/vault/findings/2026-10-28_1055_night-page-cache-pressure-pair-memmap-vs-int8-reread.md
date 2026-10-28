---
local: true
date: 2026-10-28
wave: "2026-10-28 10:05 (bench 10:55, ran 11:00 to 11:12)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl ask` / `night` / `security` (repeat 64-ask `night` sweeps when the int8 index cannot stay in page cache)"
verdict: not-viable
tags: [osai, vault, findings, ternary, pair-codes, retrieval, juniorctl, night, memmap, page-cache, memory-pressure, bench]
---

# juniorctl night under page-cache pressure: pair-code memmap against an int8 memmap that re-reads from disk

## Question

The 05:55 note ([[2026-10-28_0555_night-cold-start-load-pair-codes-vs-int8-fromfile-memmap]]) asked this next: when the int8 trit index (384 B/row) is too big to stay in page cache between `night` sweeps but the uint16 pair-code index (78 B/row) fits, does the pair-code memmap keep the repeat 64-ask sweep at least 1.5x faster than int8 memmap re-reading from disk? The 10:05 Chief run gave both pairs "no gap" (roadmap blobs unchanged), so this stays on the ask/night line.

## Method

Same corpus generator, ternarization (0.5 mean|x| absmean threshold), 64 asks, 2,048-row chunks, one BLAS `@` and per-chunk top-10 as the 10/26 to 10/28 night runs. Each run is a fresh child process with `OPENBLAS_NUM_THREADS=1` that opens the index with `np.memmap` read-only and runs several full sweeps, timing each and reading `/proc/self/io` read_bytes and major page faults per sweep. Every sweep's batch-max score matched the dense int8 reference (top-1 exact in all sweeps).

- **A, forced eviction (N = 2,000,000):** before every int8 sweep the memmap is closed, the file dropped with `posix_fadvise(DONTNEED)` and reopened, so every int8 sweep reads the full 768 MB from disk, which is what an index larger than free cache does. The pair file is never evicted. Five repeats of four sweeps; the first sweep is reported apart and the other three are the "repeat" sample (n = 15). Closing the map first matters: DONTNEED skips pages that are still mapped, which the smoke test showed as 0 bytes read.
- **B, real pressure attempt (N = 12,000,000):** int8 file 4,608,000,000 B against pair 936,000,000 B, no forced eviction, three repeats of three sweeps. MemAvailable on the shared box was 2.3 GB at the smoke test but 6.5 to 7.0 GB during this run, so the int8 file stayed cached and B measured warm-cache sweeps, not pressure.

numpy 2.5.4, 8-vCPU Xeon box with virtio disk, python3 and numpy only, no network. Index files deleted after the run. Command, in a scratch folder (not committed):

```
python3 bench.py 2000000 202610281055 5 4 1     # N, seed, repeats, sweeps per run, evict int8 before each sweep
python3 bench.py 12000000 1055 3 3 0
```

Core of the child:

```python
M={"idx":np.memmap(path,dt,'r',shape=(N,width))}
for i in range(S):
    if evict_each:                       # int8 only
        M["idx"]._mmap.close(); M.clear(); evict(path)   # unmap, then posix_fadvise DONTNEED
        M["idx"]=np.memmap(path,dt,'r',shape=(N,width))
    r0=read_bytes(); t=time.perf_counter(); top=scan(); dt=time.perf_counter()-t
```

## Results

Medians in ms (spread, max minus min, in parentheses).

| run | index | file bytes | first sweep ms | repeat sweep ms (n) | disk read per repeat | ratio int8/pair |
|---|---|---|---|---|---|---|
| A forced eviction, 2M | int8 | 768,000,000 | 2,476.1 (667.8) | 2,528.3 (945.8) (15) | 768.0 MB | |
| A forced eviction, 2M | pair | 156,000,000 | 2,984.8 (401.3) | 2,864.1 (1,129.6) (15) | 0 MB | **0.88x** |
| B 12M, cache held both | int8 | 4,608,000,000 | 15,409.9 (1,342.7) | 12,006.8 (5,885.1) (6) | 0 MB | |
| B 12M, cache held both | pair | 936,000,000 | 18,259.2 (1,423.0) | 16,885.6 (1,171.3) (6) | 0 MB | 0.71x |

The box was busier than at 05:55 (warm int8 2M sweep was about 1,490 ms then), so the absolute numbers are noisy, but the ordering held in every repeat.

## Verdict

**Not viable.** Even when every int8 sweep re-reads its whole 768 MB index from disk, the pair-code memmap that stays cached is still slower: the repeat sweep is 0.88x of int8's speed, nowhere near 1.5x faster. On this box the disk re-read costs less than the pair-table gather and cast adds to each sweep. Run B could not create real pressure because free memory rose to about 7 GB, so it only confirms the warm-cache gap (0.71x). The `night` rule stands: int8 when it fits, pair codes only when int8 cannot fit in RAM at all, and the reason is capacity (02:55 note), never speed. A slow disk (SD card or network volume around 100 MB/s, about 7.7 s per 768 MB re-read) would flip this, but that was not measured.

Caveats: synthetic tiled docs, one thread, one shared machine with noisy load; the virtio disk may be backed by a host cache, so forced-eviction re-reads may be faster than a true cold disk.

## Pair files this wave

- The 10:05 Chief gave both pairs "no gap" (22nd idle wave), so there is no pair file from this wave to link. JuniorLLM has not moved since the 05:55 note ([c1b9d31](https://github.com/cloudcover95/JuniorLLM/commit/c1b9d31)); the only other push this wave was JuniorHome 603e48f (web3node/advance_gguf_operator_sets_probe_future_named.py, left to LLC), whose path this note does not touch. The ask side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

The night line has now shown pair codes win only on capacity. Does a `security` weight-attestation pass (BETA_TO_OS Live beta step 2) hashing a ternary index stored as pair codes beat hashing the int8 index by at least 1.5x end to end (hash plus the decode needed to verify a sampled top-1), given that hashing is byte-bound rather than decode-bound?

close: b69386f2345bc522 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-28_0555)
