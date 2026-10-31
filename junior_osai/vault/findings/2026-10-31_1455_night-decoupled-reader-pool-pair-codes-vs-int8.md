---
local: true
date: 2026-10-31
wave: "2026-10-31 14:05 (bench 14:55, ran 15:00 to 15:10)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`night` sweeping the packed ask index)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, night, ask-index, reader-pool, queue, in-flight, chunk-size, device-model, bench]
---

# juniorctl night: decoupled reader pool + bounded queue vs coupled workers, uint16 pair codes vs int8 trits

## Question

The 10:55 note ([[2026-10-31_1055_night-requests-in-flight-x-chunk-size-pair-codes-vs-int8]]) found that with 8 coupled workers (each one reads a chunk, then scores it) the pair path at 400 MB/s sits near its byte floor from 4 requests in flight, 4,096-row chunks never win on a device, and above 1.6 GB/s pair codes top out at 1.32x over int8. If reads are decoupled from scoring (a separate pool of 2, 4 or 8 reader threads, each holding one request, feeding a bounded queue of 16 chunks to 8 scoring threads), does the 4,096-row chunk catch up, and does the 1.6 GB/s ceiling move past 1.5x?

## Method

Same synthetic 2,000,000-row x 384-dim index, absmean ternarization, files and device model as the 2026-10-30 14:55 script (imported from it): int8 trits on disk (384 B/row, 768,000,000 B) and uint16 pair codes (39 codes of 10 trits, 78 B/row, 156,000,000 B). Each chunk is read with `os.pread` through the modeled device: 5 ms fixed latency per request with K requests in flight sharing it (a K-slot semaphore), bytes served one request at a time at the capped rate. Page cache evicted with `posix_fadvise(DONTNEED)` before every timed sweep. OpenBLAS pinned to 1 thread, all 64 queries per chunk (B = 64), top-10. Modes: coupled 8 workers with 4 or 8 in flight (the 10:55 harness), and decoupled NR readers (in flight = NR) + 8 scorers with NR = 2, 4, 8. Grid: rates 400 and 1,600 MB/s x chunks 4,096 and 8,192 rows x 5 modes. 5 timed repeats per cell, format order reversed on odd repeats; every sweep's top-10 compared to an uncapped 8,192-row int8 reference. numpy 2.5.4, 8-vCPU shared box, MemAvailable about 4.4 GB at start, python3 and numpy only, no network, nothing bound. 100 MB/s was left out because both formats are byte-bound there (10:55 note, about 4.86x). A 131,072-row smoke run (`python3 bench.py 131072 1 1 400 5 4096 c:4,d:4`, 2.41x vs 2.40x, exact) came first. Command, run in a scratch folder (not committed):

```
python3 bench.py 2000000 202610311455 5 400,1600 5 4096,8192 c:4,c:8,d:2,d:4,d:8   # N, seed, repeats, rates MB/s, latency ms, chunks, modes
```

Core of the decoupled sweep:

```
sys.path.insert(0,"/workspace/bench1455_1030")
from bench import build, sweep, evict, Dev, topk, V10, D, PW, TOP
def sweep_dec(fp,kind,Q,NR,NS,N,mbps,lat,C,QD=16):
    dev=Dev(mbps,lat,NR); q=queue.Queue(QD)                       # NR readers -> bounded queue -> NS scorers
    def reader():  # next chunk index under a lock, dev.wait(nbytes), q.put((c,n,os.pread(fd,nb,c*C*row)))
    def scorer():  # q.get(); decode (int8 astype / np.take(V10, codes)) ; parts[c]=topk(Q@X.T,c,C)
```

## Results

Median of 5 cold 64-query sweeps over 2,000,000 rows in ms, spread (max minus min) in brackets. int8/pair above 1 means pair codes are faster. Byte floors: int8 1,920 / 480 ms and pair 390 / 98 ms at 400 / 1,600 MB/s. Latency floor is chunks x 5 ms / in flight (489 chunks at 4,096 rows, 245 at 8,192). Build took 31 s. Top-10 matched the reference exactly in every sweep.

| cap MB/s | chunk rows | mode | int8 ms | pair ms | int8/pair |
|---|---|---|---|---|---|
| 400 | 4,096 | coupled 8 workers, 4 in flight | 1,948 (3) | 782 (22) | 2.49x |
| 400 | 4,096 | coupled 8 workers, 8 in flight | 1,951 (35) | 608 (15) | 3.21x |
| 400 | 4,096 | decoupled 2 readers + 8 scorers | 2,448 (77) | 1,529 (16) | 1.60x |
| 400 | 4,096 | decoupled 4 readers + 8 scorers | 1,947 (2) | 779 (15) | 2.50x |
| 400 | 4,096 | decoupled 8 readers + 8 scorers | 1,947 (1) | 419 (3) | 4.64x |
| 400 | 8,192 | coupled 8 workers, 4 in flight | 1,941 (1) | 455 (10) | 4.27x |
| 400 | 8,192 | coupled 8 workers, 8 in flight | 1,942 (2) | 437 (18) | 4.44x |
| 400 | 8,192 | decoupled 2 readers + 8 scorers | 1,952 (58) | 890 (11) | 2.19x |
| 400 | 8,192 | decoupled 4 readers + 8 scorers | 1,943 (7) | 477 (22) | 4.07x |
| 400 | 8,192 | decoupled 8 readers + 8 scorers | 1,943 (39) | 411 (1) | 4.72x |
| 1,600 | 4,096 | coupled 8 workers, 4 in flight | 802 (25) | 702 (7) | 1.14x |
| 1,600 | 4,096 | coupled 8 workers, 8 in flight | 632 (56) | 554 (26) | 1.14x |
| 1,600 | 4,096 | decoupled 2 readers + 8 scorers | 1,725 (92) | 1,378 (26) | 1.25x |
| 1,600 | 4,096 | decoupled 4 readers + 8 scorers | 886 (71) | 707 (21) | 1.25x |
| 1,600 | 4,096 | decoupled 8 readers + 8 scorers | 531 (32) | 366 (40) | 1.45x |
| 1,600 | 8,192 | coupled 8 workers, 4 in flight | 506 (6) | 389 (28) | 1.30x |
| 1,600 | 8,192 | coupled 8 workers, 8 in flight | 511 (12) | 372 (21) | 1.37x |
| 1,600 | 8,192 | decoupled 2 readers + 8 scorers | 1,105 (55) | 739 (23) | 1.50x |
| 1,600 | 8,192 | decoupled 4 readers + 8 scorers | 577 (60) | 392 (5) | 1.47x |
| 1,600 | 8,192 | decoupled 8 readers + 8 scorers | 502 (12) | 260 (62) | 1.93x |

The coupled rows reproduce the 10:55 note within about 2 to 5 percent (400 MB/s 8,192 rows 4.27x / 4.44x vs 4.22x / 4.38x; 1.6 GB/s 1.30x / 1.37x vs 1.28x / 1.32x), so the harness is stable.

With equal requests in flight, decoupling alone does not help: 4 readers + 8 scorers matches coupled 4 in flight within noise, and 2 readers is worse than either because it halves the in-flight count. The gain comes from 8 readers + 8 scorers, which keeps 8 requests outstanding while 8 other threads score, so no worker sits idle inside a read. At 400 MB/s that brings 4,096-row chunks from 608 to 419 ms (4.64x, within about 7 percent of the 390 ms pair byte floor) and 8,192 rows from 437 to 411 ms (4.72x). 4,096-row chunks still don't beat 8,192 rows, but the gap shrinks from about 40 percent to 2 percent. At 1.6 GB/s int8 stays at about 500 to 530 ms (its decode and scoring, not reads), while pair codes at 8,192 rows drop from 372 to 260 ms, moving the ratio from 1.37x to 1.93x; the 62 ms spread on that cell means the 1.93x is the least certain number here (worst repeat about 1.7x).

## Verdict

**Viable, bounded (modeled device).** Splitting `night` into 8 reader threads and 8 scoring threads with a 16-chunk queue lifts pair codes over int8 at 1.6 GB/s and 5 ms per request from 1.37x (best coupled) to 1.93x at 8,192-row chunks (260 vs 502 ms), the first setting past 1.5x above 400 MB/s, and gives 4.64x / 4.72x at 400 MB/s for 4,096 / 8,192-row chunks. The pair path itself gets 1.43x faster than the best coupled pool at 1.6 GB/s. Practical setting: readers equal to scorers (8 + 8), 8,192-row chunks; reader count, not the queue, is what matters.

Caveats: modeled device (sleep-based latency and rate cap), not a real SD card or NVMe; synthetic embeddings, one seed, one shared 8-vCPU box; 16 threads on 8 vCPUs works only because readers mostly sleep, so a real blocking read that burns CPU would shrink the gain; the 1.93x cell has a wide spread; numpy `take` stands in for a native decode kernel.

## Pair files this wave

- No pair file for the 14:05 wave reached this run, and no new roadmap item is named; the roadmap blobs have had nothing uncovered since 2026-10-24 and the ledger will confirm the Chief's call. In JuniorLLM, the only commit in the last 6 hours is the 10:55 note [5fc44a8](https://github.com/cloudcover95/JuniorLLM/commit/5fc44a8); this note adds one new file under `junior_osai/vault/findings/` only. The ask/night side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

With 8 readers + 8 scorers, pair codes at 1.6 GB/s land at 260 ms against a 153 ms latency floor and a 98 ms byte floor, so scoring is the limit again. Does raising the reader pool past the scorer count (12 or 16 readers, 8 scorers) or batching 128 queries per chunk push the 1.6 GB/s ratio further, and does the 1.93x hold on a second seed with lower spread?

close: 80850da7b0b266d8 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-31_1055)
