---
local: true
date: 2026-11-06
wave: "2026-11-06 10:05 (bench 10:55, ran ~11:08–11:40 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` emulated eMMC/SD rate+latency+NCQ queue-depth pair vs int8; no physical media / no writable cgroup memory.high)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, ask, rate-cap, latency, ncq, queue-depth, emulated, emmc, sd, deck, sbc, cold, fadvise, dontneed, bench, b16, b32, b64, viable, speed, memory]
---

# juniorctl ask, emulated eMMC/SD rate+latency+NCQ: K=8 holds ~4.85x–4.90x; K=1 stays ≥2.3x

## Question

The 05:55 note ([[2026-11-06_0555_ask-b16-b32-b64-rate-cap-ballast-leave-target-sweep-pair-vs-int8]]) held Deck/SBC rate-cap ~4.84x–4.89x across a ballast leave-target sweep down to ~0.65–0.9 GB MemAvailable and asked for physical eMMC/SD / NCQ **or** a real writable cgroup `memory.high`. This box still has **neither** (no physical eMMC/SD; `/sys/fs/cgroup/agent/memory.high` is still not writable). This 10:55 run answers with an **emulated eMMC/SD device profile** in the existing software Dev model: under lower sustained rates + higher per-request latency + an NCQ queue-depth sweep (K ∈ {1, 8}), does Deck/SBC-style rate-capped pair-vs-int8 still hold the ~4.8x band (or at least ≥1.5x) at B ∈ {16, 32, 64}, N = 2M, exact top-10?

## Method

Synthetic 2,000,000-row × 384-dim index built with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), streamed to two flat files under the scratch dir and fsynced. Int8 file is 768,000,000 B; pair file is 156,000,000 B (4.92x smaller). Same Dev model as prior waves: K requests in flight share fixed per-request latency; bytes reserved one-at-a-time at R MB/s. Chunk size 8,192 rows; 8 worker threads; OpenBLAS / OMP pinned to 1. Scoring per chunk is int8 cast→float32 GEMM or pair `np.take(V10,…)` unpack→float32 GEMM, then chunked top-10 merge. Batches B ∈ {16, 32, 64}. 5 timed repeats per cell; format order reversed on odd repeats. Every timed batch checked against an uncapped single-thread int8 reference (exact set equality). Seed `202611061055`. Mode: **cold_rate_none only** — `posix_fadvise(..., DONTNEED)` on both bins before every timed batch; **no ballast**.

Profiles (software-emulated; no physical media):
1. **sd_like** — rates ∈ {40, 80} MB/s, lat = 15 ms, K ∈ {1, 8}
2. **emmc_like** — rate 120 MB/s, lat = 10 ms, K ∈ {1, 8}
3. **deck_anchor** — rate 100 MB/s, lat = 5 ms, K ∈ {1, 8} (confirms harness still matches prior ~4.89x at K = 8)

numpy 2.5.4, 8 vCPU shared box, MemAvailable ~6.5 GB at start / ~6.0–6.4 GB during cells, agent cgroup `memory.high` present but not writable, python3 and numpy only, no network, nothing bound except the local files. A 131,072-row smoke across all four rate:lat profiles × K ∈ {1, 8} × B ∈ {16, 32} passed first (top-10 exact). Script (`bench.py`, scratch `/workspace/bench1055_1106/`) imports `build`, `V10`, `D`, `PW`, `TOP`, `Dev` from the 2026-10-30 14:55 scratch (`/workspace/bench1455_1030/bench.py`). Commands, data bins deleted afterwards:

```
python3 bench.py 131072 202611061055 1 40:15,80:15,120:10,100:5 8 1,8 8192 16,32
python3 bench.py 2000000 202611061055 5 40:15,80:15,120:10,100:5 8 1,8 8192 16,32,64
```

## Results

Median ms per B-batch, p10 to p90 in brackets. Top-10 matched the uncapped single-thread int8 reference exactly in every timed batch. "pair vs int8" is int8 median / pair median. Observed throughput from file bytes / median wall time.

### sd_like (rate 40 / 80 MB/s, lat 15 ms, cold DONTNEED)

| rate MB/s | lat ms | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|---|---|
| 40 | 15 | 1 | 16 | 22980.7 (22971.2-22986.6) | 7674.1 (7668.6-7678.4) | **2.99x** | 33 | 20 | exact |
| 40 | 15 | 1 | 32 | 22973.6 (22970.0-22983.5) | 7671.5 (7665.6-7672.5) | **2.99x** | 33 | 20 | exact |
| 40 | 15 | 1 | 64 | 22968.5 (22964.3-22976.3) | 7671.0 (7662.1-7675.0) | **2.99x** | 33 | 20 | exact |
| 40 | 15 | 8 | 16 | 19219.8 (19219.2-19220.1) | 3920.0 (3919.8-3921.4) | **4.90x** | 40 | 40 | exact |
| 40 | 15 | 8 | 32 | 19221.6 (19221.4-19221.8) | 3922.9 (3922.4-3923.3) | **4.90x** | 40 | 40 | exact |
| 40 | 15 | 8 | 64 | 19227.2 (19226.0-19227.7) | 3929.4 (3928.2-3930.9) | **4.89x** | 40 | 40 | exact |
| 80 | 15 | 1 | 16 | 13362.1 (13352.2-13363.4) | 5712.3 (5709.9-5714.9) | **2.34x** | 57 | 27 | exact |
| 80 | 15 | 1 | 32 | 13361.6 (13354.2-13362.9) | 5711.8 (5710.1-5713.1) | **2.34x** | 57 | 27 | exact |
| 80 | 15 | 1 | 64 | 13367.5 (13363.7-13371.9) | 5723.9 (5714.5-5749.1) | **2.34x** | 57 | 27 | exact |
| 80 | 15 | 8 | 16 | 9619.8 (9619.3-9620.1) | 1971.3 (1971.2-1972.0) | **4.88x** | 80 | 79 | exact |
| 80 | 15 | 8 | 32 | 9621.9 (9621.5-9622.1) | 1974.1 (1973.7-1974.2) | **4.87x** | 80 | 79 | exact |
| 80 | 15 | 8 | 64 | 9626.1 (9625.6-9628.1) | 1980.5 (1980.3-1981.1) | **4.86x** | 80 | 79 | exact |

### emmc_like (rate 120 MB/s, lat 10 ms, cold DONTNEED)

| rate MB/s | lat ms | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|---|---|
| 120 | 10 | 1 | 16 | 8935.3 (8928.5-8936.2) | 3831.8 (3828.2-3836.7) | **2.33x** | 86 | 41 | exact |
| 120 | 10 | 1 | 32 | 8936.9 (8933.2-8937.9) | 3834.6 (3824.6-3839.9) | **2.33x** | 86 | 41 | exact |
| 120 | 10 | 1 | 64 | 8945.3 (8934.0-8946.5) | 3845.8 (3839.2-3846.8) | **2.33x** | 86 | 41 | exact |
| 120 | 10 | 8 | 16 | 6414.3 (6414.3-6414.6) | 1316.4 (1316.2-1316.5) | **4.87x** | 120 | 119 | exact |
| 120 | 10 | 8 | 32 | 6416.7 (6416.5-6416.9) | 1318.9 (1318.7-1319.2) | **4.87x** | 120 | 118 | exact |
| 120 | 10 | 8 | 64 | 6422.0 (6421.9-6422.3) | 1324.7 (1324.3-1325.8) | **4.85x** | 120 | 118 | exact |

### deck_anchor (rate 100 MB/s, lat 5 ms — prior Deck profile)

| rate MB/s | lat ms | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 5 | 1 | 16 | 8994.3 (8993.9-8996.5) | 2873.5 (2873.4-2875.4) | **3.13x** | 85 | 54 | exact |
| 100 | 5 | 1 | 32 | 8996.5 (8984.4-8996.8) | 2875.0 (2860.4-2877.2) | **3.13x** | 85 | 54 | exact |
| 100 | 5 | 1 | 64 | 9001.6 (9000.4-9003.9) | 2885.3 (2883.2-2886.8) | **3.12x** | 85 | 54 | exact |
| 100 | 5 | 8 | 16 | 7689.7 (7689.6-7689.9) | 1571.5 (1571.2-1571.5) | **4.89x** | 100 | 99 | exact |
| 100 | 5 | 8 | 32 | 7691.2 (7691.0-7691.6) | 1573.9 (1573.8-1574.0) | **4.89x** | 100 | 99 | exact |
| 100 | 5 | 8 | 64 | 7696.1 (7695.9-7697.3) | 1579.5 (1579.4-1588.8) | **4.87x** | 100 | 99 | exact |

Anchor K = 8 matches the 05:55 / 02:55 / 18:55 cold rate-cap band (~4.87x–4.89x) within noise.

## Verdict

**Viable** on both speed and memory. Under emulated eMMC/SD profiles (no physical media), **K = 8 NCQ depth** keeps pair-vs-int8 in the **~4.85x–4.90x** band across sd_like (40 / 80 MB/s @ 15 ms), emmc_like (120 MB/s @ 10 ms), and the deck_anchor (100 MB/s @ 5 ms) at B ∈ {16, 32, 64}, N = 2M, exact top-10 — matching the prior Deck/SBC cold rate-cap band and well above the 1.5x speed bar. **K = 1** (no queueing) is latency-bound and drops to **~2.33x–3.13x** (still ≥1.5x viable on speed, but no longer in the ~4.8x band); higher per-request latency hurts the smaller pair transfers relatively more when depth cannot hide it. Memory remains **4.92x smaller** (768,000,000 B vs 156,000,000 B). Answers the 05:55 Next question's physical-media half via software emulation: yes, the ~4.8x rate-cap lead holds on emulated slower / higher-latency eMMC/SD profiles **when NCQ depth ≥ 8**; at K = 1 the lead compresses but stays above 1.5x. Bounded to this synthetic / Dev-model setup on a shared 8 vCPU box with no physical eMMC/SD and no writable cgroup `memory.high`.

Caveats: synthetic embeddings; modeled Dev rate+latency+NCQ (not device NCQ / physical media / kernel block layer); N = 2M; numpy / interpreter unpack overhead; shared 8 vCPU box; one seed, five repeats per cell; profiles chosen as software stand-ins for sd_like / emmc_like, not measured from real silicon.

## Pair files this wave

- No pair file for the 10:05 wave expected (Chief both pairs no gap for many idle waves; roadmap blobs still JuniorHome ROADMAP 4efb666 / JuniorLLM BETA_TO_OS bac6923). JuniorLLM tip before this push recorded from Chief last_tips was [2164527](https://github.com/cloudcover95/JuniorLLM/commit/2164527) (05:55 vault findings). The `ask` side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Emulated eMMC/SD rate+latency+NCQ holds ~4.8x at K = 8 and ≥2.3x at K = 1. Remaining: **physical** eMMC/SD / real device NCQ when hardware exists; **or** a real writable cgroup `memory.high` (if delegation appears) that forces reclaim strictly below the int8 working set; **or** an emulated random-read / mixed R/W / thermal-throttle profile (still software Dev) if physical media stays unavailable.

close: fe70b0673b037b30 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-06_0555)
