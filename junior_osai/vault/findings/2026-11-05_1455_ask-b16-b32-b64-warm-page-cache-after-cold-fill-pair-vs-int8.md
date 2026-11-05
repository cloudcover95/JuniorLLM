---
local: true
date: 2026-11-05
wave: "2026-11-05 14:05 (bench 14:55, ran ~15:03–15:07 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` warm page-cache after cold fill pair vs int8; no physical eMMC/SD on box)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, ask, warm, page-cache, uncapped, cold-spot-check, rate-cap, deck, sbc, bench, b16, b32, b64, viable, memory]
---

# juniorctl ask, warm page-cache after cold fill: uncapped pair speed collapses; memory still 4.92x

## Question

The 10:55 note ([[2026-11-05_1055_ask-b16-b32-b64-deck-sbc-rate-cap-queue-depth-k-sweep-cold-pair-vs-int8]]) held cold Deck/SBC rate-cap × K-sweep at ~4.8x to 4.9x and asked whether physical eMMC/SD **or** host page-cache + block-layer readahead on warm runs collapses that cold lead toward/under the uncapped ~1.8x to 2.1x neighborhood. This box has no physical eMMC/SD, so this 14:55 run answers the **warm page-cache half** only: after one untimed prime fills the host page cache (no `posix_fadvise(DONTNEED)`), does warm **uncapped** (rate = 0 / no Dev wait) pair-code vs int8 stay ≥1.5x at B ∈ {16, 32, 64}, N = 2M — or does it collapse?

Modeled Dev rate-cap still throttles by bytes even when data is cached, so a warm+rate-cap arm alone cannot show page-cache bypass. Verdict basis is therefore **warm uncapped**. A short cold+rate-cap spot check (100 / 150 MB/s, K = 8, B = 16) anchors the prior ~4.8x band; one warm+rate-cap cell documents that the semaphore ignores cache (caveat only).

## Method

Synthetic 2,000,000-row × 384-dim index built with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), streamed to two flat files under the scratch dir and fsynced. Int8 file is 768,000,000 B; pair file is 156,000,000 B (4.92x smaller). Same device model as 14:55 / 05:55 / 10:55: fixed 5 ms latency per request, K requests in flight, bytes reserved one-at-a-time at R MB/s; R = 0 skips Dev entirely. Chunk size 8,192 rows; 8 worker threads; OpenBLAS / OMP pinned to 1. Scoring per chunk is int8 cast→float32 GEMM or pair `np.take(V10,…)` unpack→float32 GEMM, then chunked top-10 merge. Batches B ∈ {16, 32, 64} slice the 64-query build matrix (rotating offset per repeat). ≥5 timed repeats per cell; format order reversed on odd repeats. Every timed batch checked against an uncapped single-thread int8 reference (exact set equality).

Arms:
1. **cold_rate** spot check — `posix_fadvise(DONTNEED)` before every timed batch; rates 100 / 150 MB/s; K = 8; B = 16 only.
2. **warm_uncapped** (verdict) — one untimed sequential `pread` prime of both bins; then timed sweeps with **no** DONTNEED and rate = 0; B ∈ {16, 32, 64}.
3. **warm_rate** caveat — still cache-resident, rate = 100 MB/s, K = 8, B = 16 (no DONTNEED); expected to match cold_rate because Dev ignores cache.

numpy 2.5.4, 8 vCPU shared box, MemAvailable ~6.5 GB at start, python3 and numpy only, no network, nothing bound except the local files. A 131,072-row smoke at the same modes/rates passed first (top-10 exact). Script (`bench.py`, scratch `/workspace/bench1455_1105/`) imports `build`, `V10`, `D`, `PW`, `TOP`, `Dev` from the 2026-10-30 14:55 scratch. Commands, data bins deleted afterwards:

```
python3 bench.py 131072 202611051455 1 cold_rate,warm_uncapped,warm_rate 100,150 5 8 8 8192 16,32
python3 bench.py 2000000 202611051455 5 cold_rate,warm_uncapped,warm_rate 100,150 5 8 8 8192 16,32,64
```

## Results

Median ms per B-batch, p10 to p90 in brackets. Top-10 matched the uncapped single-thread int8 reference exactly in every timed batch. "pair vs int8" is int8 median / pair median. Observed throughput from file bytes / median wall time.

### Cold + rate-cap spot check (DONTNEED, K = 8, B = 16) — anchor

| rate MB/s | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|---|
| 100 | 8 | 16 | 7690.6 (7690.5-7691.3) | 1574.3 (1573.8-1574.9) | **4.89x** | 100 | 99 | exact |
| 150 | 8 | 16 | 5131.9 (5131.3-5136.5) | 1053.1 (1051.6-1053.4) | **4.87x** | 150 | 148 | exact |

Matches the 10:55 K = 8 B = 16 band (4.89x @ 100 / 4.87x @ 150).

### Warm uncapped (prime once, no DONTNEED, rate = 0) — verdict basis

| B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|
| 16 | 111.5 (100.0-146.5) | 131.3 (123.6-152.6) | **0.85x** | 6885 | 1188 | exact |
| 32 | 117.6 (113.8-164.2) | 141.1 (136.3-165.5) | **0.83x** | 6529 | 1105 | exact |
| 64 | 173.4 (169.2-174.7) | 192.0 (189.1-195.4) | **0.90x** | 4430 | 812 | exact |

Once both index files are cache-resident, the cold I/O-bound ~4.8x to 4.9x lead **collapses** well below the 1.5x speed bar and under the prior cold-uncapped ~1.8x to 2.1x neighborhood: pair is **slower** than int8 at every B (0.83x to 0.90x) because the `np.take(V10,…)` unpack tax is no longer hidden behind media bytes. Memory remains **4.92x smaller** (768,000,000 B vs 156,000,000 B) with exact top-10.

### Warm + rate-cap caveat (no DONTNEED, rate = 100, K = 8, B = 16)

| rate MB/s | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|---|
| 100 | 8 | 16 | 7689.4 (7689.2-7690.7) | 1573.3 (1573.0-1573.7) | **4.89x** | 100 | 99 | exact |

Identical to cold_rate within noise — Dev's byte semaphore still throttles even when pages are cached. Warm+rate-cap is therefore **not** a page-cache probe; included only as a caveat. Verdict rests solely on warm uncapped.

## Verdict

**Viable** on the memory bar (≥1.5x smaller at exact top-10): pair stays **4.92x smaller** (768 MB → 156 MB) with exact top-10 vs uncapped single-thread int8 at every warm B. Warm **speed** does **not** hold: after a cold fill primes the host page cache, uncapped pair vs int8 collapses to **0.83x / 0.85x / 0.90x** at B = 32 / 16 / 64 — below the 1.5x speed bar and under the prior cold-uncapped ~1.8x to 2.1x neighborhood (pair slower due to unpack once media is out of the path). Cold rate-cap spot check still reprints the ~4.87x to 4.89x I/O-bound band. Bounded to this synthetic / host-page-cache setup on a shared 8 vCPU box with no physical eMMC/SD; answers the 10:55 Next question's warm page-cache half: yes, warm cache-resident traffic collapses the cold speed lead.

Caveats: synthetic embeddings; host page cache (not device NCQ / physical media); N = 2M; numpy / interpreter unpack overhead; shared 8 vCPU box; one seed, five repeats per cell; warm+rate-cap equals cold because the model ignores cache.

## Pair files this wave

- No pair file for the 14:05 wave expected (Chief both pairs no gap on the 10:05 wave — 58th idle; roadmap blobs still JuniorHome ROADMAP 4efb666 / JuniorLLM BETA_TO_OS bac6923). JuniorLLM tip before this push was [502ee6d](https://github.com/cloudcover95/JuniorLLM/commit/502ee6d) (10:55 vault findings). The `ask` side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Warm host page-cache collapses the cold ~4.8x speed lead to sub-1x (pair unpack tax). Does a real physical eMMC/SD media path (NCQ / device readahead this semaphore cannot express) still hold the cold ~4.8x to 4.9x band when the working set does **not** fit in DRAM — or does block-layer readahead alone collapse it even on cold-media runs?

close: 46394d04d95653dc (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-05_1055)
