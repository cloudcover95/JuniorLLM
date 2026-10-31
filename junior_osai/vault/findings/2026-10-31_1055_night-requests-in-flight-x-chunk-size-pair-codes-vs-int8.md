---
local: true
date: 2026-10-31
wave: "2026-10-31 10:05 (bench 10:55, ran 11:02 to 11:10)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`night` sweeping the packed ask index)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, night, ask-index, queue-depth, in-flight, chunk-size, device-model, bench]
---

# juniorctl night: requests in flight (2, 4, 8) x chunk size, uint16 pair codes vs int8 trits

## Question

The 05:55 note ([[2026-10-31_0555_night-read-rate-crossover-4096-vs-8192-chunks-pair-codes-vs-int8]]) found that at 2 requests in flight the per-request latency floor (chunks x 5 ms / in flight) caps the pair path, so 4,096-row chunks lose to 8,192-row chunks on a device. How much does raising requests in flight from 2 to 4 and 8 recover, does it let the compute-best 4,096-row chunk win on storage, and how far does it move the pair-vs-int8 ratio at 100, 400 and 1,600 MB/s?

## Method

Same synthetic 2,000,000-row x 384-dim index, absmean ternarization, files and device model as the 2026-10-30 14:55 script (imported from it): int8 trits on disk (384 B/row, 768,000,000 B) and uint16 pair codes (39 codes of 10 trits, 78 B/row, 156,000,000 B). Each chunk is read with `os.pread` through a modeled device: 5 ms fixed latency per request with K requests in flight sharing it (a K-slot semaphore), bytes served one request at a time at the capped rate. Page cache evicted with `posix_fadvise(DONTNEED)` before every timed sweep. 8 worker threads, OpenBLAS pinned to 1 thread, all 64 queries per chunk (B = 64), top-10. Grid: rates 100, 400, 1,600 MB/s x in flight 2, 4, 8 x chunks 4,096 and 8,192 rows. 5 timed repeats per cell with the format order reversed on odd repeats; every sweep's top-10 compared to an uncapped 8,192-row int8 reference. numpy 2.5.4, 8-vCPU shared box, MemAvailable about 4.5 GB at start, python3 and numpy only, no network, nothing bound. A 131,072-row smoke run (`python3 bench.py 131072 1 1 400 5 8 2,8 4096`) came first. Command, run in a scratch folder (not committed):

```
python3 bench.py 2000000 202610311055 5 100,400,1600 5 8 2,4,8 4096,8192   # N, seed, repeats, rates MB/s, latency ms, threads, in flight list, chunks
```

Core of the script:

```
sys.path.insert(0,"/workspace/bench1455_1030")
from bench import build, sweep, evict                  # the 2026-10-30 14:55 script (device model + pread sweep)
for mb in RATES:
    for K in (2,4,8):
        for C in (4096, 8192):
            for r in range(R):
                for kind in (("int8","pair") if r%2==0 else ("pair","int8")):
                    evict(fp); t=time.perf_counter(); I=sweep(fp,kind,Q,8,N,mb,5,C,K)
                    res[kind].append(time.perf_counter()-t); ok&=np.array_equal(I,ref)
```

## Results

Median of 5 cold 64-query sweeps over 2,000,000 rows in ms, spread (max minus min) in brackets. int8/pair above 1 means pair codes are faster. Latency floor is chunks x 5 ms / in flight. Byte floors at each rate: int8 7,680 / 1,920 / 480 ms and pair 1,560 / 390 / 98 ms at 100 / 400 / 1,600 MB/s. Build took 35 s. Top-10 matched the reference exactly in every sweep.

| cap MB/s | in flight | chunk rows | int8 ms | pair ms | int8/pair | latency floor ms |
|---|---|---|---|---|---|---|
| 100 | 2 | 4,096 | 7,710 (20) | 2,126 (67) | 3.63x | 1,222 |
| 100 | 2 | 8,192 | 7,700 (9) | 1,585 (4) | 4.86x | 612 |
| 100 | 4 | 4,096 | 7,708 (1) | 1,589 (1) | 4.85x | 611 |
| 100 | 4 | 8,192 | 7,700 (1) | 1,582 (1) | 4.87x | 306 |
| 100 | 8 | 4,096 | 7,707 (3) | 1,609 (78) | 4.79x | 306 |
| 100 | 8 | 8,192 | 7,700 (2) | 1,584 (9) | 4.86x | 153 |
| 400 | 2 | 4,096 | 2,292 (24) | 1,512 (29) | 1.52x | 1,222 |
| 400 | 2 | 8,192 | 1,948 (67) | 872 (39) | 2.23x | 612 |
| 400 | 4 | 4,096 | 1,949 (5) | 779 (18) | 2.50x | 611 |
| 400 | 4 | 8,192 | 1,943 (3) | 460 (15) | 4.22x | 306 |
| 400 | 8 | 4,096 | 1,948 (6) | 599 (27) | 3.25x | 306 |
| 400 | 8 | 8,192 | 1,944 (95) | 444 (28) | 4.38x | 153 |
| 1,600 | 2 | 4,096 | 1,571 (13) | 1,382 (15) | 1.14x | 1,222 |
| 1,600 | 2 | 8,192 | 916 (8) | 722 (14) | 1.27x | 612 |
| 1,600 | 4 | 4,096 | 804 (41) | 702 (5) | 1.15x | 611 |
| 1,600 | 4 | 8,192 | 530 (22) | 413 (11) | 1.28x | 306 |
| 1,600 | 8 | 4,096 | 641 (28) | 550 (13) | 1.17x | 306 |
| 1,600 | 8 | 8,192 | 533 (69) | 402 (36) | 1.32x | 153 |

The 2-in-flight rows reproduce the 05:55 note within about 1 percent (3.63x / 4.86x at 100 MB/s, 1.52x / 2.23x at 400 MB/s, 1.14x / 1.27x at 1.6 GB/s), so the harness is stable.

At 400 MB/s, int8 is pinned to its byte floor (about 1,945 ms) as soon as there are 4 requests in flight, while the pair path was latency-bound. Going from 2 to 4 in flight cuts pair time at 8,192-row chunks from 872 to 460 ms, and the ratio goes from 2.23x to 4.22x; 8 in flight adds a little more (444 ms, 4.38x, within about 15 percent of the 390 ms pair byte floor). At 4,096-row chunks the pair path also improves (1,512 to 779 to 599 ms) but stays behind 8,192 rows at every queue depth, because at 8 workers there are never more than 8 requests outstanding and each extra request still costs its 5 ms. So raising queue depth does not make the compute-best 4,096-row chunk win on storage.

At 100 MB/s both formats are byte-bound from 4 in flight on, and the ratio settles at the byte ratio (about 4.85x, close to the 4.92x size ratio). At 1.6 GB/s pair codes are compute-bound (about 400 ms at 8,192 rows) and int8 hits its own floor around 530 ms, so neither queue depth nor chunk size gets past 1.32x.

## Verdict

**Viable, bounded, on mid-speed storage (modeled device).** At 400 MB/s with 5 ms per request, raising requests in flight from 2 to 4 with 8,192-row chunks makes pair codes 4.22x faster than int8 (460 vs 1,943 ms), up from 2.23x, and 4.38x at 8 in flight; at 100 MB/s the ratio holds at about 4.86x regardless. Above about 1.6 GB/s pair codes stay below 1.5x (1.32x at best). Practical setting for `night` reading from storage: 8,192-row chunks with at least 4 requests in flight; 4,096 rows is never better on a device at 8 workers.

Caveats: modeled device (sleep-based latency and rate cap), not a real SD card or NVMe; synthetic embeddings, one seed, one shared 8-vCPU box; numpy `take` stands in for a native decode kernel; in flight is capped by 8 worker threads, so 8 in flight is the most this harness can express.

## Pair files this wave

- No pair file for the 10:05 wave reached this run, and no new roadmap item is named; the roadmap blobs have had nothing uncovered since 2026-10-24 and the ledger will confirm the Chief's call. In JuniorLLM, the only commit in the last 6 hours is the 05:55 note [6a287f5](https://github.com/cloudcover95/JuniorLLM/commit/6a287f5); this note adds one new file under `junior_osai/vault/findings/` only. The ask/night side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

With 4 or more requests in flight the pair path at 400 MB/s sits within about 15 percent of its byte floor, so the remaining gap is decode compute. Does decoupling reads from scoring (a separate reader pool of 2 to 4 threads feeding a bounded queue to 8 scoring workers) let 4,096-row chunks or more queries per batch close that gap, and does it change the 1.6 GB/s ceiling of 1.32x?

close: f6713ec9541860d2 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-31_0555)
