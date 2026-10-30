---
local: true
date: 2026-10-30
wave: "2026-10-30 14:05 (bench 14:55, ran 15:00 to 15:03)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`night` sweeping the packed ask index)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, night, ask-index, read-rate, worker-threads, decode-bound, bench]
---

# juniorctl night: read rate and worker threads at 8,192-row chunks and 2 in flight, int8 trits vs uint16 pair codes

## Question

The 10:55 note ([[2026-10-30_1055_night-queue-depth-chunk-size-pair-codes-vs-int8]]) found pair codes 4.86x faster than int8 at 100 MB/s with 8,192-row chunks and 2 requests in flight. Is the night sweep still I/O-bound on faster storage? Rerunning that cell at 400 MB/s and 1.6 GB/s with 2, 4 and 8 worker threads, where does the advantage fall toward the 1.29x uncapped figure from 02:55, and what becomes the bottleneck?

## Method

Same index, encoding and sweep as the 02:55, 05:55 and 10:55 runs: 2,000,000 unique rows, 384-dim, ternarized with the 0.5 mean|x| absmean rule, stored as int8 trits (768,000,000 bytes) and as uint16 pair codes (156,000,000 bytes, 4.92x smaller). 64 queries, top-10, OpenBLAS pinned to 1 thread, page cache evicted before every timed sweep. Pair codes are decoded with the 59,049 x 10 int8 table gather.

Fixed: 8,192-row chunks (245 requests per sweep for either format), 2 requests in flight, 5 ms per request, using the 10:55 device model (a K-slot semaphore around a fixed latency, then bytes streamed one request at a time at the capped rate). Varied: read cap 400 MB/s, 1,600 MB/s and uncapped (no latency, real disk with cold cache), and 2, 4 and 8 worker threads. Each cell has 3 repeats with alternating int8 and pair order; every timed sweep's top-10 was checked against a 1-thread uncapped int8 reference. The MB/s columns are file bytes divided by median sweep time.

numpy 2.5.4, 8-vCPU shared box, MemAvailable about 2.2 GB at start (lower than earlier runs), python3 and numpy only, no network, nothing bound. Index files were deleted afterwards. A 100,000-row smoke run (`python3 bench.py 100000 1 1 400,0 5 2,8 2 8192`) came first. Command, in a scratch folder (not committed):

```
python3 bench.py 2000000 202610301455 3 400,1600,0 5 2,4,8 2 8192     # N, seed, repeats, read caps MB/s (0 = uncapped), latency ms, worker threads, in flight, chunk rows
```

The script is the 10:55 script with its `__main__` loop replaced by a read-cap x thread-count grid at one in-flight depth and one chunk size (same build, encoding, device model, sweep and eviction code; 10:55's tracemalloc table dropped):

```
    RATES=[int(x) for x in sys.argv[4].split(",")]; TS=[int(x) for x in sys.argv[6].split(",")]; K=int(sys.argv[7]); C=int(sys.argv[8])
    ...
    for mb in RATES:
        for T in TS:
            res={"int8":[],"pair":[]}; ok=True
            for r in range(R):
                for kind in (("int8","pair") if r%2==0 else ("pair","int8")):
                    evict(fp); t=time.perf_counter(); I=sweep(fp,kind,Q,T,N,mb,(lat if mb>0 else 0),C,K); res[kind].append(time.perf_counter()-t); ok&=np.array_equal(I,ref)
            a,b=res["int8"],res["pair"]; ma,mb_=np.median(a),np.median(b)
            print(f"{mb if mb else 'uncapped'} | {T} | {ma*1e3:.0f} ({np.ptp(a)*1e3:.0f}) | {mb_*1e3:.0f} ({np.ptp(b)*1e3:.0f}) | {ma/mb_:.2f}x | {os.path.getsize(fp['int8'])/ma/1e6:.0f} | {os.path.getsize(fp['pair'])/mb_/1e6:.0f} | {'exact' if ok else 'MISMATCH'}",flush=True)
```

## Results

Median of 3 sweeps in ms, spread (max minus min) in brackets. Build took 34 s.

| read cap | threads | int8 ms | pair codes ms | int8/pair | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|
| 400 MB/s | 2 | 2413 (44) | 1559 (68) | 1.55x | 318 | 100 | exact |
| 400 MB/s | 4 | 1942 (1) | 873 (11) | **2.22x** | 395 | 179 | exact |
| 400 MB/s | 8 | 1943 (1) | 873 (5) | **2.23x** | 395 | 179 | exact |
| 1.6 GB/s | 2 | 1688 (46) | 1397 (76) | 1.21x | 455 | 112 | exact |
| 1.6 GB/s | 4 | 934 (28) | 757 (57) | 1.23x | 822 | 206 | exact |
| 1.6 GB/s | 8 | 919 (8) | 720 (6) | 1.28x | 836 | 217 | exact |
| uncapped | 2 | 775 (61) | 675 (17) | 1.15x | 991 | 231 | exact |
| uncapped | 4 | 434 (27) | 373 (21) | 1.16x | 1772 | 418 | exact |
| uncapped | 8 | 298 (18) | 212 (16) | 1.41x | 2576 | 737 | exact |

At 400 MB/s, int8 hits the byte limit (395 MB/s, 1.94 s) from 4 threads up, while pair codes finish in 0.87 s, so pair codes stay 2.22x faster. Pair codes are not byte-bound there: 245 requests at 5 ms with 2 in flight is a floor of about 0.61 s before any bytes move, so their remaining cost is request latency, not decode.

At 1.6 GB/s the advantage collapses to 1.21x to 1.28x. int8 no longer reaches its 0.48 s byte floor (it lands at 0.92 s), and pair codes sit at 0.72 to 0.76 s, near the same latency floor. Both formats are now paying mostly for request latency and per-chunk compute, which bytes do not change, so the 4.92x size gap stops mattering.

With 2 threads, decode and matmul become the bottleneck for both formats: uncapped pair codes need 675 ms at 2 threads vs 212 ms at 8, and at 2 threads the ratio is 1.15x to 1.55x at every read cap. The pair-code table gather is not cheaper per row than the int8 cast, so once reads are hidden there is little left for the smaller format to win.

## Verdict

**Viable, bounded.** On storage at or below about 400 MB/s (SD cards and USB sticks, the Pi target), pair codes keep a 2.22x sweep advantage over int8 with 4 or more worker threads and exact top-10. On 1.6 GB/s storage (NVMe class) the gain falls to 1.21x to 1.28x and is not worth the decode table there; with only 2 worker threads it is not worth it at any rate above 100 MB/s except 400 MB/s, where it is borderline at 1.55x.

Rule for `night`: use pair codes with 8,192-row chunks when the index sits on SD or USB storage and at least 4 worker threads are available; on NVMe, int8 is within 1.3x and simpler. Thread count matters more than format once storage is fast: going from 2 to 8 threads cuts the int8 sweep 1.8x at 1.6 GB/s.

Caveats: the device model is in-process and was not measured on a real Pi, SD card, USB stick or NVMe drive. One latency (5 ms, which is high for NVMe and makes the 1.6 GB/s cells latency-heavy), one depth (2), synthetic embeddings, one seed, 3 repeats, a shared machine with only about 2.2 GB available at start.

## Pair files this wave

- No pair file was produced for the 14:05 wave: Quant Chief assigned "no gap" to both pairs (roadmap blobs unchanged at 4efb666/bac6923, no LLC skip paths). In JuniorLLM, the only commit in the last 6 hours is the 10:55 note itself ([e78fc7f](https://github.com/cloudcover95/JuniorLLM/commit/e78fc7f)), so no Trit Pack, Handshake or LLC path was touched. This note adds one new file under `junior_osai/vault/findings/` only. The ask/night side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

On fast storage the sweep is per-chunk compute and latency bound. Does fusing the decode into the score (scoring each uint16 pair code with a per-query 59,049-entry lookup of partial dot products, so no int8 or float32 row is ever built) beat the int8 cast-and-matmul path when uncapped at 2, 4 and 8 threads, and at what query batch size does the lookup table stop fitting in cache?

close: 4c41b371cb6f1c5d (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-30_1055)
