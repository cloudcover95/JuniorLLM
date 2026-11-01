---
local: true
date: 2026-10-31
wave: "2026-10-31 18:05 (bench 18:55, ran 19:02 to 19:08)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`night` sweeping the packed ask index)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, night, ask-index, reader-pool, queue, replication, device-model, bench]
---

# juniorctl night: reader pool past the scorer count (8 / 12 / 16 readers, 8 scorers), second seed, uint16 pair codes vs int8 trits

## Question

The 14:55 note ([[2026-10-31_1455_night-decoupled-reader-pool-pair-codes-vs-int8]]) found that 8 reader threads feeding 8 scoring threads through a 16-chunk queue lift pair codes over int8 at 1.6 GB/s from 1.37x to 1.93x at 8,192-row chunks, but that cell had a 62 ms spread and ran on one seed. Does the 8 + 8 result hold on a second seed, does it carry to a faster 3.2 GB/s device, and does raising readers past scorers (12 or 16 readers, 8 scorers) push the ratio further?

## Method

Same synthetic 2,000,000-row x 384-dim index, absmean ternarization, files, device model and decoupled sweep as the 14:55 script (imported from it unchanged), new seed 202610311855. int8 trits on disk are 384 B/row (768,000,000 B); uint16 pair codes are 39 codes of 10 trits, 78 B/row (156,000,000 B). Each chunk is read with `os.pread` through the modeled device: 5 ms fixed latency per request with K requests in flight sharing it, bytes served one request at a time at the capped rate. Page cache evicted with `posix_fadvise(DONTNEED)` before every timed sweep. OpenBLAS pinned to 1 thread, 64 queries per chunk (B = 64), top-10, 8,192-row chunks (245 chunks). Modes: coupled 8 workers with 8 in flight (baseline), and decoupled NR readers (in flight = NR) + 8 scorers with NR = 8, 12, 16. Rates 1,600 and 3,200 MB/s. 5 timed repeats per cell, format order reversed on odd repeats; every sweep's top-10 compared to an uncapped int8 reference. numpy 2.5.4, 8-vCPU shared box, MemAvailable about 4.2 GB at start, python3 and numpy only, no network, nothing bound. A 131,072-row smoke run came first (`python3 bench.py 131072 7 1 1600 5 8192 c:8,d:8,d:16`, exact top-10; too small to time). Command, run in a scratch folder (not committed); `bench.py` is the 14:55 script with its import path pointed at the 14:55 and 2026-10-30 14:55 scratch folders:

```
python3 bench.py 2000000 202610311855 5 1600,3200 5 8192 c:8,d:8,d:12,d:16   # N, seed, repeats, rates MB/s, latency ms, chunks, modes
```

## Results

Median of 5 cold 64-query sweeps over 2,000,000 rows in ms, spread (max minus min) in brackets. int8/pair above 1 means pair codes are faster. Byte floors: int8 480 / 240 ms and pair 98 / 49 ms at 1,600 / 3,200 MB/s. Latency floor is 245 chunks x 5 ms / in flight: 153 / 102 / 77 ms for 8 / 12 / 16. Build took 35 s. Top-10 matched the reference exactly in every sweep.

| cap MB/s | mode | int8 ms | pair ms | int8/pair |
|---|---|---|---|---|
| 1,600 | coupled 8 workers, 8 in flight | 592 (117) | 412 (25) | 1.43x |
| 1,600 | decoupled 8 readers + 8 scorers | 586 (167) | 272 (38) | 2.16x |
| 1,600 | decoupled 12 readers + 8 scorers | 538 (42) | 285 (47) | 1.89x |
| 1,600 | decoupled 16 readers + 8 scorers | 526 (59) | 298 (38) | 1.76x |
| 3,200 | coupled 8 workers, 8 in flight | 532 (46) | 408 (34) | 1.30x |
| 3,200 | decoupled 8 readers + 8 scorers | 426 (78) | 258 (35) | 1.65x |
| 3,200 | decoupled 12 readers + 8 scorers | 348 (72) | 257 (24) | 1.35x |
| 3,200 | decoupled 16 readers + 8 scorers | 342 (41) | 263 (16) | 1.30x |

On the new seed, 8 readers + 8 scorers at 1.6 GB/s gives pair codes 272 ms (14:55: 260 ms) against int8 586 ms, 2.16x (14:55: 1.93x); the pair spread narrows from 62 to 38 ms, so the result replicates. The spreads bound the worst case: pair at most 310 ms and int8 at least 419 ms, so no pairing of repeats falls below 1.35x. The coupled baseline reproduces too (1.43x vs 1.37x). At 3.2 GB/s the same 8 + 8 split still clears the bar at 1.65x (258 vs 426 ms).

Raising readers past scorers does not help pair codes: 272, 285 and 298 ms at 1.6 GB/s and 258, 257 and 263 ms at 3.2 GB/s for 8, 12 and 16 readers. Pair codes are scoring-bound at about 260 ms once 8 requests are in flight (the latency floor is 153 ms and the byte floor under 100 ms), and extra reader threads only add contention on 8 vCPUs. int8, which still has read time to hide, does gain from more readers (586 to 526 ms at 1.6 GB/s, 426 to 342 ms at 3.2 GB/s), so the ratio falls to 1.76x and 1.30x. The int8 spreads (up to 167 ms) are wider than the pair spreads in every decoupled cell.

## Verdict

**Viable, bounded (modeled device), and the reader-past-scorer idea is not worth it.** The 8 reader + 8 scorer `night` layout keeps pair codes 2.16x faster than int8 at 1.6 GB/s (272 vs 586 ms) on a second seed with tighter spread, and 1.65x at 3.2 GB/s (258 vs 426 ms), at exact top-10 and 4.9x fewer bytes. 12 or 16 readers with 8 scorers make pair codes slightly slower and int8 faster, so they shrink the gain to 1.30x to 1.89x; keep readers equal to scorers. The pair path is now limited by decode plus scoring (about 260 ms), not storage.

Caveats: modeled device (sleep-based latency and rate cap), not a real SD card or NVMe; synthetic embeddings, two seeds total across the 14:55 and 18:55 runs; shared 8-vCPU box with wide int8 spreads; up to 24 threads on 8 vCPUs; numpy `take` stands in for a native decode kernel.

## Pair files this wave

- No pair file for the 18:05 wave reached this run, and no new roadmap item is named; the roadmap blobs have had nothing uncovered since 2026-10-24 and the ledger will confirm the Chief's call. In JuniorLLM the only commit in the last 6 hours is the 14:55 note [defb05a](https://github.com/cloudcover95/JuniorLLM/commit/defb05a); this note adds one new file under `junior_osai/vault/findings/` only. The ask/night side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

With storage hidden, pair codes sit at about 260 ms of decode plus scoring for 2,000,000 rows x 64 queries. Does scoring straight from the pair codes (a per-query 59,049-entry pair-score table summed over 39 codes, no float unpack) or batching 128 queries per chunk cut that compute floor enough to lift the 3.2 GB/s ratio past 2x?

close: 6b43a365293fc64e (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-31_1455)
