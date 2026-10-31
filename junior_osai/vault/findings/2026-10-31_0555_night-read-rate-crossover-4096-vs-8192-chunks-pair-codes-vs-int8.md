---
local: true
date: 2026-10-31
wave: "2026-10-31 05:05 (bench 05:55, ran 06:00 to 06:05)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`night` sweeping the packed ask index)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, night, ask-index, chunk-size, read-rate, device-model, bench]
---

# juniorctl night: read-rate crossover at 4,096 vs 8,192-row chunks, uint16 pair codes vs int8 trits

## Question

The 02:55 note ([[2026-10-31_0255_night-chunk-size-sweep-b64-pair-decode-vs-int8]]) found that in memory, 4,096-row chunks at 8 workers are the fastest setting for both formats, and that pair-code decode is 10 to 17 percent slower than int8 on compute alone. Once `night` reads the index from storage, does the compute-best 4,096-row chunk keep the pair format's byte advantage, and at what sustained read rate (100 MB/s to uncapped) do pair codes stop beating int8 by 1.5x, compared with the 8,192-row crossover from the 14:55 note ([[2026-10-30_1455_night-read-rate-threads-pair-codes-vs-int8]])?

## Method

Same synthetic 2,000,000-row x 384-dim index, absmean ternarization, files and device model as the 14:55 script (imported from it): int8 trits on disk (384 B/row, 768,000,000 B) and uint16 pair codes (39 codes of 10 trits, 78 B/row, 156,000,000 B). Each chunk is read with `os.pread` through a modeled device: 5 ms fixed latency per request, 2 requests in flight, bytes served one request at a time at the capped rate (0 means uncapped). Page cache evicted with `posix_fadvise(DONTNEED)` before every timed sweep. 8 worker threads, OpenBLAS pinned to 1 thread, all 64 queries per chunk (B = 64), top-10. Chunk sizes 4,096 and 8,192 rows; rates 100, 400 and 1,600 MB/s and uncapped. 5 timed repeats per cell with the format order reversed on odd repeats; every sweep's top-10 compared to an uncapped 8,192-row int8 reference. numpy 2.5.4, 8-vCPU shared box, MemAvailable about 2.4 GB at start, python3 and numpy only, no network, nothing bound. A 131,072-row smoke run (`python3 bench.py 131072 1 1 400,0 5 8 2 4096,8192`) came first. Command, run in a scratch folder (not committed):

```
python3 bench.py 2000000 202610310555 5 100,400,1600,0 5 8 2 4096,8192   # N, seed, repeats, rates MB/s, latency ms, threads, in flight, chunks
```

Core of the script:

```
sys.path.insert(0,"/workspace/bench1455_1030")
from bench import build, sweep, evict                  # the 2026-10-30 14:55 script (device model + pread sweep)
for mb in RATES:
    for C in (4096, 8192):
        for r in range(R):
            for kind in (("int8","pair") if r%2==0 else ("pair","int8")):
                evict(fp); t=time.perf_counter(); I=sweep(fp,kind,Q,8,N,mb,(5 if mb>0 else 0),C,2)
                res[kind].append(time.perf_counter()-t); ok&=np.array_equal(I,ref)
```

## Results

Median of 5 cold 64-query sweeps over 2,000,000 rows in ms, spread (max minus min) in brackets. int8/pair above 1 means pair codes are faster. Effective MB/s is file bytes over median sweep time. Build took 35 s. Top-10 matched the reference exactly in every sweep.

| cap MB/s | chunk rows | int8 ms | pair ms | int8/pair | int8 MB/s | pair MB/s |
|---|---|---|---|---|---|---|
| 100 | 4,096 | 7,709 (2) | 2,127 (5) | 3.62x | 100 | 73 |
| 100 | 8,192 | 7,701 (9) | 1,583 (34) | 4.87x | 100 | 99 |
| 400 | 4,096 | 2,308 (13) | 1,528 (9) | 1.51x | 333 | 102 |
| 400 | 8,192 | 1,944 (9) | 873 (36) | 2.23x | 395 | 179 |
| 1,600 | 4,096 | 1,571 (22) | 1,381 (12) | 1.14x | 489 | 113 |
| 1,600 | 8,192 | 915 (19) | 729 (14) | 1.26x | 840 | 214 |
| uncapped | 4,096 | 345 (25) | 235 (17) | 1.47x | 2,229 | 664 |
| uncapped | 8,192 | 309 (26) | 229 (25) | 1.35x | 2,485 | 681 |

The 8,192-row rows reproduce the earlier notes closely (4.87x at 100 MB/s as on 2026-10-30 02:55, 2.23x at 400 MB/s and 1.26x at 1.6 GB/s as at 14:55), so the harness is stable across runs.

The compute-best 4,096-row chunk is the wrong choice once there is a per-request latency. With 489 chunks instead of 245, the 5 ms latency at 2 in flight alone costs about 489 x 5 / 2 = 1,222 ms per sweep, and the pair path sits right on that floor (1,381 to 1,528 ms at 400 and 1,600 MB/s). That cuts the pair advantage from 2.23x to 1.51x at 400 MB/s and from 4.87x to 3.62x at 100 MB/s, and makes 4,096-row chunks slower than 8,192-row chunks for both formats on the device (int8 2,308 vs 1,944 ms at 400 MB/s). Only uncapped, with no latency, does 4,096 rows edge ahead (1.47x vs 1.35x), matching the 02:55 in-memory result.

The 1.5x crossover sits just above 400 MB/s at 4,096-row chunks (1.51x there, 1.14x at 1.6 GB/s) and somewhere between 400 MB/s and 1.6 GB/s at 8,192-row chunks (2.23x vs 1.26x), so the larger chunk keeps the pair format viable over a wider range of storage speeds.

## Verdict

**Viable, bounded, on slow storage (modeled device).** At 100 MB/s with 5 ms per request, pair codes are 4.87x faster than int8 at 8,192-row chunks and 3.62x at 4,096; at 400 MB/s, 2.23x and 1.51x. Above about 1.6 GB/s neither chunk size reaches 1.5x. The new result is about the chunk size: the 4,096-row compute optimum from 02:55 should not be used when `night` reads from a device with per-request latency, because the extra requests eat most of the pair format's 4.92x byte saving. Practical setting: 8,192-row chunks when reading from storage, 4,096 rows only when the index is already resident.

Caveats: modeled device (sleep-based latency and rate cap), not a real SD card or NVMe; synthetic embeddings, one seed, one shared 8-vCPU box; numpy `take` stands in for a native decode kernel. The 8,192-row numbers repeat earlier notes and are not a new gain on their own.

## Pair files this wave

- No pair file for the 05:05 wave reached this run, and no new roadmap item is named; the roadmap blobs have had nothing uncovered since 2026-10-24, and the ledger will confirm the Chief's call. In JuniorLLM, the commits in the last 6 hours are the 02:55 note [1f17deb](https://github.com/cloudcover95/JuniorLLM/commit/1f17deb) and [40f749d](https://github.com/cloudcover95/JuniorLLM/commit/40f749d) (02:52 MT, cloudcover95, `rails/nopull/no_model_pull_check.py`, unattributed to a pair in this run); this note does not touch `rails/` and adds one new file under `junior_osai/vault/findings/` only. The ask/night side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Since the latency floor (chunks x latency / in flight) is what limits the pair path at small chunks, how much does raising requests in flight from 2 to 4 and 8 recover at 4,096-row chunks and 400 MB/s, and does that let the compute-best chunk size win on storage too, or does 8,192 rows stay ahead?

close: 5b2410629ddb0b06 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-31_0255)
