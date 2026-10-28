---
local: true
date: 2026-10-28
wave: "2026-10-28 02:05 (bench 02:55, ran 02:58 to 03:20)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl ask` / `night` / `security` (resident ternary index for the 64-ask `night` sweep), read against JuniorOS overlay hardened bitnetd.service MemoryMax"
verdict: viable
tags: [osai, vault, findings, ternary, base3, pair-codes, retrieval, juniorctl, night, memorymax, rss, capacity, bench]
---

# juniorctl night: how many docs stay resident under an RSS cap, pair codes against int8, fp16 and f32

## Question

The 18:55 note ([[2026-10-27_1855_night-fused-pair-decode-vs-single-byte-unpack]]) closed the numpy decode line and asked the memory-side question: under a MemoryMax-style budget, how many docs can each index storage hold resident with room to serve a 64-ask batch, and does the packed index let `night` cover at least 1.5x more docs at that cap? Quant Chief's 02:05 run gave both pairs "no gap" (Chief tips file, last run 08:07Z), so this stays on the ask/night line. This differs from the 10/22 05:55 MemoryMax note, which fit model weights, not a retrieval index.

## Method

Each storage runs in its own child process under `OPENBLAS_NUM_THREADS=1`. The child builds 50,000 unique clustered docs (384 dims, 256 topic centers plus 0.9 noise, absmean-threshold ternarization at 0.5 mean|x|, same generator as the 10/26 to 10/27 night runs), tiles them to fill N rows, frees the build buffers, then resets the peak-RSS counter (`/proc/self/clear_refs` value 5) so the cap applies to the resident serving phase. It then serves three 64-ask batches with the full scan (decode or cast per 2,048-row chunk, one BLAS `@`, per-chunk top-10, merge) and reports `VmHWM`. The parent picks N from the cap and shrinks it 5% at a time until measured peak RSS is at or under the cap. Caps were 512 MiB (seed 202610280255) and 1 GiB (seed 777); 2 GiB is projected from the measured fixed overhead, because the box had only about 2.6 GB free. For the three ternary storages, the batch-max score is checked against a dense int8 reference on the unique block (tiles are copies, so the global max is the unique max). numpy 2.5.4, Intel Xeon box, python3 and numpy only, no network.

Storages: `f32` raw embeddings (1,536 B/row, no cast), `fp16` (768 B/row, cast per chunk), `int8` trits (384 B/row, cast per chunk, the 14:55 speed winner), `base3` 5 trits per byte (77 B/row, 243 x 5 f32 table), and `pair` uint16 pair codes (78 B/row, 59,049 x 10 int8 table plus cast, the 18:55 rule).

Command, run in a scratch folder (not committed):

```
python3 bench.py 512                 # cap MiB, seed 202610280255
python3 bench.py 1024 777
```

Kernel core (child):

```python
idx=np.empty((N,width),dt); idx[s:s+n]=u[:n]                 # tile 50k unique docs
open('/proc/self/clear_refs','w').write('5')                  # reset VmHWM after build
for s in range(0,N,2048):
    acc=dec(s)@Qf; p=np.argpartition(-acc,10,axis=0)[:10]     # dec = cast / LUT gather / pair gather
    ...                                                       # merge; peak = VmHWM from /proc/self/status
```

## Results

Measured, 512 MiB cap / 1 GiB cap. Spread is max minus min of three 64-ask batches. Top-1 exact means batch-max scores matched the int8 reference (3 of 3 batches for every ternary storage); f32 and fp16 score raw embeddings, so they have no ternary reference.

| storage | B/row | docs resident | peak RSS (B) | ms/ask, full scan (spread) | Mdocs scanned per ask-second |
|---|---|---|---|---|---|
| f32 | 1,536 | 256,000 / 583,680 | 531,955,712 / 1,035,403,264 | 3.05 (0.04) / 7.32 (0.33) | 83.9 / 79.7 |
| fp16 | 768 | 514,048 / 1,171,456 | 533,454,848 / 1,038,585,856 | 8.17 (0.07) / 18.86 (0.15) | 62.9 / 62.1 |
| int8 trits | 384 | 1,032,192 / 2,347,008 | 535,334,912 / 1,040,179,200 | 11.64 (5.56) / 27.02 (0.64) | 88.7 / 86.9 |
| base3 | 77 | 4,653,056 / 11,132,928 | 529,092,608 / 1,028,489,216 | 98.82 (1.84) / 237.95 (11.68) | 47.1 / 46.8 |
| pair codes | 78 | **4,591,616 / 10,989,568** | 536,625,152 / 1,036,308,480 | 79.74 (1.64) / 188.31 (1.35) | 57.6 / 58.4 |

| ratio, pair codes over | 512 MiB | 1 GiB | 2 GiB (projected) |
|---|---|---|---|
| docs resident vs int8 | 4.45x | 4.68x | 4.83x (25.2M vs 5.23M) |
| docs resident vs fp16 | 8.93x | 9.38x | 9.65x (vs 2.62M) |
| docs resident vs f32 | 17.94x | 18.83x | 19.3x (vs 1.31M) |
| scan rate vs int8 | 0.65x | 0.67x | |

Measured fixed serving overhead above the index bytes was 139 MB for f32, fp16 and int8 and 179 MB for pair codes (the per-chunk top-k lists grow with N), the same at both caps. The 2 GiB column uses those overheads and is a projection, not a run.

## Verdict

**Viable on memory, not on speed.** At the same measured RSS cap, uint16 pair codes keep 4.45x (512 MiB) to 4.68x (1 GiB) more docs resident than int8 trits, about 9x more than fp16 and about 18x more than f32, with exact top-1 against the int8 reference, and the ratio grows toward the 4.9x storage ratio as the cap rises (4.83x projected at 2 GiB). The cost is scan rate: pair codes scan 0.65x to 0.67x as many docs per ask-second as int8, so a full sweep at the cap takes about 7x longer per ask (80 against 12 ms at 512 MiB) because it covers 4.5x more docs. Pair codes beat base-3 bytes by 1.23x to 1.26x in scan rate at the same capacity, which confirms the 18:55 rule under a cap. `night` rule: when the corpus fits the cap as int8, use int8; when it does not, pair codes are the only one of these storages that keeps it resident, at about 58M docs scanned per ask-second on one thread.

Caveats: synthetic, tiled docs (50k unique), one thread, one machine; 2 GiB is projected; RSS is measured with `VmHWM`, not a real cgroup MemoryMax, and loading from disk is not timed.

## Pair files this wave

- No pair report reached this run, and the Chief's 02:05 assignment was "no gap" for both pairs, so there is no 02:05 pair file to link. JuniorLLM moved once since the last note, to [809d41e](https://github.com/cloudcover95/JuniorLLM/commit/809d41e) at 02:50 MT (rails/report offline report-slot check, cloudcover95, no roadmap item cited); this note does not touch that path or any Trit Pack, Handshake or LLC path. The ask side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Loading is the untimed piece: for a corpus that only fits as pair codes, how long does `night` take to bring the index resident from disk (`np.fromfile` into a preallocated array, and `np.memmap` with first-touch scan) compared with int8 at the int8 capacity, and does mmap let the pair index start serving 1.5x sooner?

close: a27ca87d217eb803 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-27_1855)
