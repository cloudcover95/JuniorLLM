---
local: true
date: 2026-11-05
wave: "2026-11-05 18:05 (bench 18:55, ran ~18:58–19:11 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` cold rate-cap block-layer readahead fadvise pair vs int8; no physical eMMC/SD on box)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, ask, cold, rate-cap, readahead, fadvise, sequential, willneed, dontneed, deck, sbc, bench, b16, b32, b64, viable, speed, memory]
---

# juniorctl ask, cold rate-cap + block-layer readahead fadvise: ~4.8x speed holds under SEQUENTIAL / WILLNEED

## Question

The 14:55 note ([[2026-11-05_1455_ask-b16-b32-b64-warm-page-cache-after-cold-fill-pair-vs-int8]]) showed warm host page-cache collapsing cold ~4.8x speed to 0.83x–0.90x (pair unpack tax once media is out of the path) while memory stayed 4.92x, and asked whether physical eMMC/SD **or** block-layer readahead on cold-media collapses that cold lead. This box still has no physical eMMC/SD, so this 18:55 run answers the **block-layer readahead half** only: with the working set forced out of DRAM via `POSIX_FADV_DONTNEED` before every timed batch, does readahead advice alone — `POSIX_FADV_SEQUENTIAL` on the whole file and/or `POSIX_FADV_WILLNEED` ahead-window before each chunk — collapse the cold Deck/SBC rate-capped pair-vs-int8 lead toward/under 1.5x at B ∈ {16, 32, 64}, N = 2M, rates 100 / 150 MB/s, K = 8 — or does the ~4.8x to 4.9x band hold?

## Method

Synthetic 2,000,000-row × 384-dim index built with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), streamed to two flat files under the scratch dir and fsynced. Int8 file is 768,000,000 B; pair file is 156,000,000 B (4.92x smaller). Same device model as 14:55 / 10:55 / 05:55: fixed 5 ms latency per request, K = 8 requests in flight, bytes reserved one-at-a-time at R MB/s; R = 0 skips Dev entirely. Chunk size 8,192 rows; 8 worker threads; OpenBLAS / OMP pinned to 1. Scoring per chunk is int8 cast→float32 GEMM or pair `np.take(V10,…)` unpack→float32 GEMM, then chunked top-10 merge. Batches B ∈ {16, 32, 64} slice the 64-query build matrix (rotating offset per repeat). ≥5 timed repeats per cell; format order reversed on odd repeats. Every timed batch checked against an uncapped single-thread int8 reference (exact set equality). Seed `202611051855`.

Arms (all cold: `posix_fadvise(..., DONTNEED)` on both bins **before every timed batch** so the working set does not stay in DRAM):
1. **cold_rate_none** (anchor) — open + `pread` only; no readahead advice. Expect ~4.87x–4.89x like 14:55 / 10:55.
2. **cold_rate_seq** — after open, `POSIX_FADV_SEQUENTIAL` on the whole file (still DONTNEED between batches).
3. **cold_rate_willneed** — after open, before each chunk read, `POSIX_FADV_WILLNEED` for the upcoming ahead-window (next 4 chunks / ~32,768 rows worth of bytes) in addition to the current read; still DONTNEED between batches.
4. **cold_uncapped** spot (context only) — DONTNEED + rate = 0, B = 16; not the verdict basis (rate-cap is the Deck/SBC story).

numpy 2.5.4, 8 vCPU shared box, MemAvailable ~6.3 GB at start, python3 and numpy only, no network, nothing bound except the local files. A 131,072-row smoke at the same modes/rates passed first (top-10 exact). Script (`bench.py`, scratch `/workspace/bench1855_1105/`) imports `build`, `V10`, `D`, `PW`, `TOP`, `Dev` from the 2026-10-30 14:55 scratch (`/workspace/bench1455_1030/bench.py`). Commands, data bins deleted afterwards:

```
python3 bench.py 131072 202611051855 1 cold_rate_none,cold_rate_seq,cold_rate_willneed,cold_uncapped 100,150 5 8 8 8192 16,32
python3 bench.py 2000000 202611051855 5 cold_rate_none,cold_rate_seq,cold_rate_willneed,cold_uncapped 100,150 5 8 8 8192 16,32,64
```

## Results

Median ms per B-batch, p10 to p90 in brackets. Top-10 matched the uncapped single-thread int8 reference exactly in every timed batch. "pair vs int8" is int8 median / pair median. Observed throughput from file bytes / median wall time.

### cold_rate_none (DONTNEED, no advice, K = 8) — anchor

| rate MB/s | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|---|
| 100 | 8 | 16 | 7690.4 (7689.8-7692.8) | 1571.9 (1571.8-1574.3) | **4.89x** | 100 | 99 | exact |
| 100 | 8 | 32 | 7692.8 (7692.5-7693.3) | 1575.2 (1575.1-1576.2) | **4.88x** | 100 | 99 | exact |
| 100 | 8 | 64 | 7698.7 (7698.5-7699.8) | 1581.0 (1580.4-1581.3) | **4.87x** | 100 | 99 | exact |
| 150 | 8 | 16 | 5130.5 (5130.4-5131.8) | 1052.0 (1051.8-1052.4) | **4.88x** | 150 | 148 | exact |
| 150 | 8 | 32 | 5134.1 (5133.6-5134.7) | 1055.0 (1053.7-1055.3) | **4.87x** | 150 | 148 | exact |
| 150 | 8 | 64 | 5139.8 (5139.0-5140.2) | 1061.0 (1060.6-1061.2) | **4.84x** | 149 | 147 | exact |

Matches the 14:55 / 10:55 K = 8 cold rate-cap band (~4.84x–4.89x).

### cold_rate_seq (DONTNEED + POSIX_FADV_SEQUENTIAL, K = 8)

| rate MB/s | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|---|
| 100 | 8 | 16 | 7690.7 (7690.3-7692.3) | 1572.4 (1572.0-1572.7) | **4.89x** | 100 | 99 | exact |
| 100 | 8 | 32 | 7692.9 (7692.6-7693.0) | 1575.6 (1575.4-1576.0) | **4.88x** | 100 | 99 | exact |
| 100 | 8 | 64 | 7700.6 (7699.7-7702.8) | 1583.7 (1582.0-1584.6) | **4.86x** | 100 | 99 | exact |
| 150 | 8 | 16 | 5132.0 (5131.0-5141.6) | 1053.2 (1052.7-1053.5) | **4.87x** | 150 | 148 | exact |
| 150 | 8 | 32 | 5134.8 (5134.3-5291.2) | 1056.2 (1055.2-1058.7) | **4.86x** | 150 | 148 | exact |
| 150 | 8 | 64 | 5141.1 (5140.5-5141.7) | 1062.2 (1061.3-1066.5) | **4.84x** | 149 | 147 | exact |

Indistinguishable from `cold_rate_none` within noise — SEQUENTIAL advice does **not** collapse the cold rate-cap lead.

### cold_rate_willneed (DONTNEED + POSIX_FADV_WILLNEED ahead-window ×4 chunks, K = 8)

| rate MB/s | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|---|
| 100 | 8 | 16 | 7691.0 (7690.7-7693.8) | 1572.6 (1571.9-1574.5) | **4.89x** | 100 | 99 | exact |
| 100 | 8 | 32 | 7694.7 (7693.8-7696.8) | 1576.1 (1574.9-1577.8) | **4.88x** | 100 | 99 | exact |
| 100 | 8 | 64 | 7699.6 (7699.2-7701.4) | 1582.6 (1581.4-1583.3) | **4.87x** | 100 | 99 | exact |
| 150 | 8 | 16 | 5131.4 (5130.9-5134.6) | 1052.9 (1052.6-1053.9) | **4.87x** | 150 | 148 | exact |
| 150 | 8 | 32 | 5135.9 (5134.7-5136.4) | 1055.4 (1055.4-1056.0) | **4.87x** | 150 | 148 | exact |
| 150 | 8 | 64 | 5140.9 (5140.5-5142.8) | 1061.9 (1061.5-1062.3) | **4.84x** | 149 | 147 | exact |

Also indistinguishable from the anchor — WILLNEED ahead-window does **not** collapse the cold rate-cap lead either.

### cold_uncapped spot (DONTNEED, rate = 0, B = 16) — context only

| B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|
| 16 | 261.6 (259.8-270.9) | 138.5 (134.3-144.6) | **1.89x** | 2935 | 1126 | exact |

Prior cold-uncapped ~1.8x–2.1x neighborhood reprinted; not the Deck/SBC verdict basis.

## Verdict

**Viable** on both speed and memory. Block-layer readahead advice alone does **not** collapse the cold Deck/SBC rate-capped pair-vs-int8 lead: under forced DONTNEED between batches, `cold_rate_none` / `cold_rate_seq` / `cold_rate_willneed` all stay in the **~4.84x–4.89x** band at rates 100 / 150 MB/s, K = 8, B ∈ {16, 32, 64} (exact top-10) — well above the 1.5x speed bar and matching the 14:55 / 10:55 cold anchors. Memory remains **4.92x smaller** (768,000,000 B vs 156,000,000 B). The *readahead question's speed half* therefore does **not** collapse (speed holds under SEQUENTIAL and WILLNEED); contrast with 14:55 where warm page-cache *did* collapse uncapped speed to 0.83x–0.90x. Bounded to this synthetic / host-fadvise setup on a shared 8 vCPU box with no physical eMMC/SD (advice is a hint the kernel / block layer may ignore under modeled Dev byte throttling); answers the 14:55 Next question's readahead half: no, fadvise SEQUENTIAL / WILLNEED alone does not collapse the cold rate-cap lead when DONTNEED keeps the working set out of DRAM.

Caveats: synthetic embeddings; modeled Dev rate-cap (not device NCQ / physical media); host `posix_fadvise` is advisory and DONTNEED between batches defeats any long-lived readahead state; N = 2M; numpy / interpreter unpack overhead; shared 8 vCPU box; one seed, five repeats per cell; ahead-window fixed at 4 chunks.

## Pair files this wave

- No pair file for the 18:05 wave expected (Chief both pairs no gap — 60th idle; roadmap blobs still JuniorHome ROADMAP 4efb666 / JuniorLLM BETA_TO_OS bac6923). JuniorLLM tip before this push was [82f9629](https://github.com/cloudcover95/JuniorLLM/commit/82f9629) (14:55 vault findings). The `ask` side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Readahead fadvise alone does not collapse cold rate-cap ~4.8x when DONTNEED keeps the working set out of DRAM. Remaining physical-media / NCQ half when hardware exists: does a real eMMC/SD path (device readahead / NCQ this semaphore cannot express) still hold the cold ~4.8x–4.9x band when the working set does **not** fit in DRAM — or does cgroup `memory.high` eviction under the same rate-cap collapse it toward the warm-cache sub-1x regime?

close: e1c5bc406d5d2558 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-05_1455)
