---
local: true
date: 2026-10-28
wave: "2026-10-28 05:05 (bench 05:55, ran 06:02 to 06:12)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl ask` / `night` / `security` (cold start of the resident ternary index for the 64-ask `night` sweep)"
verdict: not-viable
tags: [osai, vault, findings, ternary, pair-codes, retrieval, juniorctl, night, cold-start, memmap, page-cache, bench]
---

# juniorctl night cold start: loading pair codes from disk against int8 trits

## Question

The 02:55 note ([[2026-10-28_0255_night-resident-capacity-under-rss-cap-pair-codes-vs-int8-fp16-f32]]) left loading untimed. For the same corpus, how long does `night` take to bring the index resident from disk and finish its first 64-ask sweep when the index is stored as uint16 pair codes (78 B/row) instead of int8 trits (384 B/row), using either a read into a preallocated array or `np.memmap` with first-touch scan? Does the pair index start serving at least 1.5x sooner? No 05:05 Chief assignment or pair report reached this run (the recent waves have all been "no gap"), so this stays on the ask/night line.

## Method

Both index files hold the same N docs (50,000 unique clustered 384-dim docs, tiled; same generator and 0.5 mean|x| absmean ternarization as the 10/26 to 10/28 night runs), written and fsynced to the box disk. Before each "cold" run the file is dropped from the page cache with `posix_fadvise(DONTNEED)`; before each "warm" run it is read once in full. Each run is a fresh child process with `OPENBLAS_NUM_THREADS=1` that (1) loads the index, either `readinto` a preallocated `np.empty` array (`fromfile`) or `np.memmap` read-only, then (2) runs the full 64-ask sweep (decode or cast per 2,048-row chunk, one BLAS `@`, per-chunk top-10), then (3) runs a second sweep for the warm scan time. "Ready" is load plus first sweep. Every run's batch-max score matched the dense int8 reference on the unique block (top-1 exact in all 80 runs). Five repeats per cell, median and spread (max minus min). numpy 2.5.4, 8-vCPU Xeon box with virtio disk, python3 and numpy only, no network. The index files were deleted after the run.

Command, run in a scratch folder (not committed):

```
python3 bench.py 2000000 202610280555 5     # N docs, seed, repeats
python3 bench.py 1000000 555 5
```

Core of the child:

```python
t0=time.perf_counter()
if mode=="fromfile":
    idx=np.empty((N,width),dt)
    with open(path,'rb') as f: f.readinto(memoryview(idx).cast('B'))
else: idx=np.memmap(path,dt,'r',shape=(N,width))
t1=time.perf_counter()
for s in range(0,N,2048):                       # dec = int8 cast, or 59,049 x 10 pair-table gather + cast
    acc=dec(s)@Qf; p=np.argpartition(-acc,10,axis=0)[:10] ...
t2=time.perf_counter()                          # ready = t2 - t0
```

## Results

Medians over five runs in ms (spread in parentheses), N = 2,000,000 docs (int8 file 768,000,000 B, pair file 156,000,000 B).

| storage | load | cache | load ms | first sweep ms | ready ms | warm sweep ms |
|---|---|---|---|---|---|---|
| int8 | read | cold | 710.0 (240.0) | 1,498.5 (93.0) | 2,222.0 (223.2) | 1,486.1 (146.2) |
| int8 | read | warm | 238.7 (37.3) | 1,454.8 (44.6) | 1,713.9 (55.8) | 1,443.7 (74.2) |
| int8 | memmap | cold | 0.2 | 2,046.5 (168.0) | 2,046.8 (168.0) | 1,494.8 (156.4) |
| int8 | memmap | warm | 0.3 | 1,534.9 (113.3) | 1,535.1 (113.3) | 1,520.8 (224.3) |
| pair | read | cold | 101.6 (60.0) | 2,191.8 (164.6) | 2,293.4 (214.9) | 2,202.1 (148.9) |
| pair | read | warm | 46.4 (47.0) | 2,249.7 (123.9) | 2,296.1 (123.5) | 2,227.1 (191.2) |
| pair | memmap | cold | 0.3 | 2,285.7 (65.7) | 2,286.0 (65.7) | 2,265.1 (291.8) |
| pair | memmap | warm | 0.3 | 2,207.9 (78.8) | 2,208.2 (78.8) | 2,195.7 (63.9) |

Ratios, int8 time over pair time (above 1 means pair codes are faster), at 2M docs / 1M docs (1M run: int8 file 384,000,000 B, pair 78,000,000 B):

| path | load only | ready (load + first sweep) |
|---|---|---|
| read, cold cache | **6.99x / 8.68x** | 0.97x / 0.96x |
| read, warm cache | 5.14x / 7.57x | 0.75x / 0.83x |
| memmap, cold cache | 0.86x / 1.04x (both under 1 ms) | 0.90x / 0.89x |
| memmap, warm cache | 0.99x / 0.96x | 0.70x / 0.68x |

Cold first-touch through memmap added about 550 ms to the int8 first sweep (2,047 against 1,495 ms warm) and about 20 to 90 ms to the pair sweep, which is the disk saving showing up inside the scan instead of before it.

## Verdict

**Not viable for cold start.** Reading pair codes off disk is 7.0x to 8.7x faster than reading int8 trits (4.9x fewer bytes), but on this box a cold int8 read of 768 MB takes only about 0.7 s, while the pair-code sweep is about 0.7 s slower than the int8 sweep at 2M docs. The two cancel: time until the first 64-ask batch is answered is 0.96x to 0.97x for pair codes on a cold cache (slightly slower) and 0.68x to 0.83x on a warm cache, never near 1.5x sooner. memmap does not change the order; it only moves the disk time into the first sweep. `night` rule stands from the 02:55 note: use int8 when the corpus fits the cap, and pair codes only when it does not, where the reason is capacity, not start-up time. The load-only gain would matter on a slower disk (an SD card or network volume at around 100 MB/s would make the int8 read about 7.7 s against 1.6 s), but that was not measured here.

Caveats: synthetic tiled docs, one thread, one machine; "cold" means evicted from the page cache with `posix_fadvise`, and the virtio disk may be backed by a host cache, so true cold disk reads could be slower and would favor pair codes more.

## Pair files this wave

- No 05:05 pair report or Chief assignment reached this run, so there is no pair file from this wave to link. JuniorLLM has not moved since the 02:55 note ([71c4091](https://github.com/cloudcover95/JuniorLLM/commit/71c4091)); the most recent other JuniorLLM push is [809d41e](https://github.com/cloudcover95/JuniorLLM/commit/809d41e) (rails/report, 02:50 MT), whose path this note does not touch. The ask side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Under memory pressure, an int8 memmap index larger than free page cache gets evicted between sweeps. When the int8 file is bigger than available page cache but the pair-code file fits, does a pair-code memmap keep repeat-sweep latency for `night` at least 1.5x lower than int8 memmap re-reading from disk?

close: e1885c50f79bb8ff (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-28_0255)
