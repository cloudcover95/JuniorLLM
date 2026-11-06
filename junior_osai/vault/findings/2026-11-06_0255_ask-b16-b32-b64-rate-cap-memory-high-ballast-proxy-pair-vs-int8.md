---
local: true
date: 2026-11-06
wave: "2026-11-06 02:05 (bench 02:55, ran ~03:05–03:35 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` rate-cap memory.high-ballast-proxy pair vs int8; no writable cgroup memory.high / no physical eMMC/SD)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, ask, rate-cap, memory-high, ballast, proxy, reclaim, deck, sbc, bench, b16, b32, b64, viable, speed, memory]
---

# juniorctl ask, rate-cap + memory.high ballast-proxy: ~4.8x speed holds under anonymous reclaim pressure

## Question

The 18:55 note ([[2026-11-05_1855_ask-b16-b32-b64-cold-rate-cap-block-readahead-fadvise-pair-vs-int8]]) held cold Deck/SBC rate-cap ~4.84x–4.89x under fadvise SEQUENTIAL / WILLNEED and asked whether physical eMMC/SD / NCQ **or** cgroup `memory.high` eviction under the same rate-cap collapses that lead toward the warm-uncapped 0.83x–0.90x regime. This box still has no physical eMMC/SD, and `/sys/fs/cgroup/agent/memory.high` exists but is **not writable** by `box` (Permission denied; not escalated). This 02:55 run answers the **eviction half** with a **user-space anonymous-memory ballast** proxy: under Deck/SBC rate-cap (rates 100 / 150 MB/s, K = 8, lat 5 ms, chunk 8192, 8 threads), does a concurrent ballast that drives MemAvailable below / near the int8 working set (forcing reclaim / thrashing as a `memory.high` proxy) collapse pair-vs-int8 toward the warm-uncapped 0.83x–0.90x regime at B ∈ {16, 32, 64}, N = 2M — or does the cold ~4.8x–4.9x band hold?

## Method

Synthetic 2,000,000-row × 384-dim index built with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), streamed to two flat files under the scratch dir and fsynced. Int8 file is 768,000,000 B; pair file is 156,000,000 B (4.92x smaller). Same device model as 18:55 / 14:55 / 10:55: fixed 5 ms latency per request, K = 8 requests in flight, bytes reserved one-at-a-time at R MB/s; R = 0 skips Dev entirely. Chunk size 8,192 rows; 8 worker threads; OpenBLAS / OMP pinned to 1. Scoring per chunk is int8 cast→float32 GEMM or pair `np.take(V10,…)` unpack→float32 GEMM, then chunked top-10 merge. Batches B ∈ {16, 32, 64} slice the 64-query build matrix (rotating offset per repeat). ≥5 timed repeats per cell; format order reversed on odd repeats. Every timed batch checked against an uncapped single-thread int8 reference (exact set equality). Seed `202611060255`.

Arms (all under the same Dev rate-cap model):
1. **cold_rate_none** (anchor) — `posix_fadvise(..., DONTNEED)` on both bins **before every timed batch**; no ballast. Expect ~4.84x–4.89x.
2. **rate_ballast** — **no** DONTNEED between batches; start a child ballast process (`mmap` anonymous + touch every page; try `mlock` — ulimit locked memory only 8 MB so locked_mb stayed 0) sized to leave MemAvailable roughly below the int8 file (target leave 700 MB; hard cap 9.5–10 GB); optionally warm-read both files once after ballast starts; then timed rate-capped batches. Ballast status / MemAvailable recorded per cell. A top-up recheck kept MemAvailable ~1.3–1.4 GB throughout (ballast ≈ 10,048 MB).
3. Spots (context only, not verdict basis): **warm_uncapped_ballast** at B = 16 under the same ballast; **cold_uncapped** at B = 16 after ballast stop.

numpy 2.5.4, 8 vCPU shared box, MemAvailable ~6–11 GB at start (≈10 GB before ballast), agent cgroup `memory.high` present but not writable, python3 and numpy only, no network, nothing bound except the local files / ballast child. A 131,072-row smoke at the same modes/rates passed first (top-10 exact). Script (`bench.py` + `ballast_worker.py`, scratch `/workspace/bench0255_1106/`) imports `build`, `V10`, `D`, `PW`, `TOP`, `Dev` from the 2026-10-30 14:55 scratch (`/workspace/bench1455_1030/bench.py`). Commands, data bins deleted afterwards:

```
python3 bench.py 131072 202611060255 1 cold_rate_none,rate_ballast,cold_uncapped 100,150 5 8 8 8192 16,32 40 800
python3 bench.py 2000000 202611060255 5 cold_rate_none,rate_ballast,warm_uncapped,cold_uncapped 100,150 5 8 8 8192 16,32,64 700 9500
# top-up recheck (sustained MemAvailable ~1.3–1.4 GB):
python3 bench.py 2000000 202611060255 5 rate_ballast,warm_uncapped,cold_uncapped 100,150 5 8 8 8192 16,32,64 700 10000
```

## Results

Median ms per B-batch, p10 to p90 in brackets. Top-10 matched the uncapped single-thread int8 reference exactly in every timed batch. "pair vs int8" is int8 median / pair median. Observed throughput from file bytes / median wall time. Primary tables from the main N = 2M run (ballast 9,280 MB → MemAvailable ≈ 714 MB at start, ~2.1–2.3 GB during cells); top-up recheck (ballast 10,048 MB, MemAvailable held ~1.37–1.44 GB across all rate_ballast cells) reprinted the same speed band within noise.

### cold_rate_none (DONTNEED, no ballast, K = 8) — anchor

| rate MB/s | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|---|
| 100 | 8 | 16 | 7691.4 (7690.1-7692.0) | 1572.7 (1572.4-1573.1) | **4.89x** | 100 | 99 | exact |
| 100 | 8 | 32 | 7694.2 (7694.2-7695.4) | 1576.2 (1575.4-1577.0) | **4.88x** | 100 | 99 | exact |
| 100 | 8 | 64 | 7700.3 (7699.9-7701.5) | 1582.2 (1582.0-1583.1) | **4.87x** | 100 | 99 | exact |
| 150 | 8 | 16 | 5131.9 (5130.7-5132.5) | 1052.8 (1052.0-1053.2) | **4.87x** | 150 | 148 | exact |
| 150 | 8 | 32 | 5135.7 (5134.7-5136.2) | 1055.8 (1054.9-1057.0) | **4.86x** | 150 | 148 | exact |
| 150 | 8 | 64 | 5141.6 (5140.6-5143.0) | 1063.4 (1062.7-1064.9) | **4.84x** | 149 | 147 | exact |

Matches the 18:55 / 14:55 / 10:55 K = 8 cold rate-cap band (~4.84x–4.89x).

### rate_ballast (no DONTNEED + anonymous ballast, K = 8) — memory.high proxy

Main run (ballast ≈ 9,280 MB; MemAvailable during cells ≈ 2.1–2.3 GB):

| rate MB/s | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 | MemAvail MB |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 8 | 16 | 7691.2 (7690.2-7691.6) | 1573.0 (1572.8-1574.8) | **4.89x** | 100 | 99 | exact | 2246 |
| 100 | 8 | 32 | 7692.1 (7691.7-7693.9) | 1575.6 (1574.7-1576.4) | **4.88x** | 100 | 99 | exact | 2127 |
| 100 | 8 | 64 | 7699.2 (7698.6-7700.4) | 1581.7 (1581.2-1583.6) | **4.87x** | 100 | 99 | exact | 2131 |
| 150 | 8 | 16 | 5130.9 (5130.6-5137.3) | 1054.1 (1052.9-1054.3) | **4.87x** | 150 | 148 | exact | 2128 |
| 150 | 8 | 32 | 5133.6 (5133.5-5134.0) | 1055.6 (1055.4-1055.9) | **4.86x** | 150 | 148 | exact | 2299 |
| 150 | 8 | 64 | 5140.0 (5139.8-5142.0) | 1062.3 (1062.0-1062.5) | **4.84x** | 149 | 147 | exact | 2140 |

Top-up recheck (ballast ≈ 10,048 MB; MemAvailable held ≈ 1.37–1.44 GB across cells — still above the 924 MB combined working set, but the tightest sustained pressure this box allowed without OOM):

| rate MB/s | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 | MemAvail MB |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 8 | 16 | 7690.5 (7689.8-7690.7) | 1573.0 (1572.3-1573.2) | **4.89x** | 100 | 99 | exact | 1368 |
| 100 | 8 | 32 | 7693.4 (7692.2-7694.0) | 1575.7 (1575.1-1577.1) | **4.88x** | 100 | 99 | exact | 1440 |
| 100 | 8 | 64 | 7701.7 (7699.6-7702.3) | 1583.5 (1582.8-1585.7) | **4.86x** | 100 | 99 | exact | 1378 |
| 150 | 8 | 16 | 5131.1 (5130.7-5131.3) | 1053.1 (1052.9-1053.7) | **4.87x** | 150 | 148 | exact | 1372 |
| 150 | 8 | 32 | 5134.3 (5133.4-5134.8) | 1056.4 (1056.2-1057.3) | **4.86x** | 150 | 148 | exact | 1375 |
| 150 | 8 | 64 | 5140.2 (5139.7-5141.1) | 1062.5 (1062.2-1063.8) | **4.84x** | 149 | 147 | exact | 1378 |

Indistinguishable from `cold_rate_none` within noise — the memory.high-ballast proxy does **not** collapse the rate-cap lead.

### warm_uncapped_ballast spot (no DONTNEED, rate = 0, B = 16, under ballast) — context only

| run | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 | MemAvail MB |
|---|---|---|---|---|---|---|---|---|
| main | 16 | 91.9 (90.1-102.3) | 111.7 (111.1-115.2) | **0.82x** | 8357 | 1397 | exact | 2138 |
| top-up | 16 | 90.4 (88.6-95.9) | 108.3 (108.1-117.1) | **0.83x** | 8495 | 1440 | exact | 1378 |

Warm-collapse neighborhood (0.82x–0.83x) reprinted under ballast when uncapped — working set still fits in the residual MemAvailable, so page-cache residency + unpack tax dominate once Dev is removed. Not the Deck/SBC verdict basis.

### cold_uncapped spot (DONTNEED, rate = 0, B = 16, ballast stopped) — context only

| B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|
| 16 | 420.2 (418.2-483.0) | 159.9 (145.1-205.7) | **2.63x** | 1828 | 976 | exact |

Prior cold-uncapped ~1.8x–2.6x neighborhood; not the Deck/SBC verdict basis.

## Verdict

**Viable** on both speed and memory. User-space anonymous ballast as a `memory.high` proxy does **not** collapse the Deck/SBC rate-capped pair-vs-int8 lead: under no-DONTNEED + ballast (≈ 9.3–10.0 GB anonymous, MemAvailable driven to ≈ 0.7 GB at start and held ≈ 1.3–2.3 GB during timed cells), `rate_ballast` stays in the **~4.84x–4.89x** band at rates 100 / 150 MB/s, K = 8, B ∈ {16, 32, 64} (exact top-10) — matching `cold_rate_none` and well above the 1.5x speed bar. Memory remains **4.92x smaller** (768,000,000 B vs 156,000,000 B). The *eviction question's speed half under rate-cap* therefore does **not** collapse toward the warm-uncapped 0.83x–0.90x regime; that sub-1x signature appears only on the uncapped-under-ballast spot (0.82x–0.83x), consistent with the 14:55 warm page-cache finding once Dev byte-throttle is removed. Bounded to this synthetic / host-ballast setup on a shared 8 vCPU box with no writable cgroup `memory.high` and no physical eMMC/SD (`mlock` capped at 8 MB so ballast was touch-retained anonymous mmap, not locked); answers the 18:55 Next question's eviction half via proxy: no, rate-cap ~4.8x holds under anonymous reclaim pressure.

Caveats: synthetic embeddings; modeled Dev rate-cap (not device NCQ / physical media); ballast is a user-space proxy not a real cgroup `memory.high` reclaim path; MemAvailable during timed cells stayed ≈ 1.3–2.3 GB (above the 924 MB combined working set), so full thrashing-below-WS was not achieved without risking host OOM; N = 2M; numpy / interpreter unpack overhead; shared 8 vCPU box; one seed, five repeats per cell.

## Pair files this wave

- No pair file for the 02:05 wave expected (Chief both pairs no gap for many idle waves; roadmap blobs still JuniorHome ROADMAP 4efb666 / JuniorLLM BETA_TO_OS bac6923). JuniorLLM tip before this push recorded from the prior vault clone was [013c005](https://github.com/cloudcover95/JuniorLLM/commit/013c005) (18:55 vault findings); fresh clone tip recorded at push. The `ask` side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Ballast-proxy eviction under rate-cap does not collapse ~4.8x when MemAvailable stays just above the working set. Remaining: physical eMMC/SD / NCQ when hardware exists; **or** a real writable cgroup `memory.high` (if delegation appears) that forces reclaim strictly below the int8 working set; **or** a ballast-size sweep that deliberately drives MemAvailable under pair-only vs int8-only footprints to see whether asymmetric reclaim ever breaks the rate-cap ratio.

close: b9162b72759ea183 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-05_1855)
