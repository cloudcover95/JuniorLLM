---
local: true
date: 2026-11-06
wave: "2026-11-06 14:05 (bench 14:55, ran ~14:59–15:24 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` emulated random-read / mixed R/W / thermal-throttle pair vs int8; no physical media / no writable cgroup memory.high)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, ask, rate-cap, latency, ncq, queue-depth, emulated, random-read, mixed-rw, thermal-throttle, deck, sbc, cold, fadvise, dontneed, bench, b16, b32, b64, viable, speed, memory]
---

# juniorctl ask, emulated random-read / mixed R/W / thermal: K=8 holds ~4.82x–4.91x

## Question

The 10:55 note ([[2026-11-06_1055_ask-b16-b32-b64-emulated-emmc-sd-rate-latency-ncq-pair-vs-int8]]) held emulated eMMC/SD rate+latency+NCQ ~4.85x–4.90x at K = 8 (≥2.3x at K = 1) and asked for physical eMMC/SD / real writable cgroup `memory.high` **or** emulated random-read / mixed R/W / thermal-throttle if hardware stayed unavailable. This box still has **neither** physical media (only loop*/vda/vdb) nor a writable `/sys/fs/cgroup/agent/memory.high` (value `max`). This 14:55 run answers with three **emulated Dev stress profiles**: under (1) random/scattered chunk reads (higher effective latency), (2) mixed R/W contention (read bandwidth halved by synthetic write traffic), and (3) thermal-throttle (rate steps down mid-batch then recovers), does Deck/SBC-style rate-capped pair-vs-int8 still hold ≥1.5x (and ideally the ~4.8x band at K = 8) at B ∈ {16, 32, 64}, N = 2M, exact top-10?

## Method

Synthetic 2,000,000-row × 384-dim index built with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), streamed to two flat files under the scratch dir and fsynced. Int8 file is 768,000,000 B; pair file is 156,000,000 B (4.92x smaller). Same Dev model as prior waves: K requests in flight share fixed per-request latency; bytes reserved one-at-a-time at R MB/s. Thermal profile extends Dev with a progress-fraction rate schedule (bytes-served / total). Chunk size 8,192 rows; 8 worker threads; OpenBLAS / OMP pinned to 1. Scoring per chunk is int8 cast→float32 GEMM or pair `np.take(V10,…)` unpack→float32 GEMM, then chunked top-10 merge. Batches B ∈ {16, 32, 64}. 5 timed repeats per cell; format order reversed on odd repeats. Every timed batch checked against an uncapped single-thread int8 reference (exact set equality). Seed `202611061455`. Mode: **cold_rate_none only** — `posix_fadvise(..., DONTNEED)` on both bins before every timed batch; **no ballast**.

Profiles (software-emulated; no physical media):
1. **deck_anchor** — rate 100 MB/s, lat = 5 ms, K = 8 (confirms harness still matches prior ~4.87x–4.89x)
2. **random_read** — rate 100 MB/s, lat = 25 ms (random-seek penalty), K ∈ {1, 8}
3. **mixed_rw** — effective read rate 50 MB/s (half of 100; synthetic write contention), lat = 5 ms, K ∈ {1, 8}
4. **thermal** — mid-batch rate schedule 100 → 40 → 80 MB/s by progress thirds (first / middle / last third of bytes), lat = 5 ms, K = 8 (`ThermalDev` extends Dev)

numpy 2.5.4, 8 vCPU shared box, MemAvailable ~4.4 GB at start / ~6.0–6.4 GB during cells (bins + refs), agent cgroup `memory.high` present but not writable, python3 and numpy only, no network, nothing bound except the local files. A 131,072-row smoke across all four profiles × listed K × B ∈ {16, 32} passed first (top-10 exact). Script (`bench.py`, scratch `/workspace/bench1455_1106/`) imports `build`, `V10`, `D`, `PW`, `TOP`, `Dev` from the 2026-10-30 14:55 scratch (`/workspace/bench1455_1030/bench.py`) and defines local `ThermalDev`. Commands, data bins deleted afterwards:

```
python3 bench.py 131072 202611061455 1 8 8192 16,32
python3 bench.py 2000000 202611061455 5 8 8192 16,32,64
```

## Results

Median ms per B-batch, p10 to p90 in brackets. Top-10 matched the uncapped single-thread int8 reference exactly in every timed batch. "pair vs int8" is int8 median / pair median. Observed throughput from file bytes / median wall time.

### deck_anchor (rate 100 MB/s, lat 5 ms, K = 8 — prior Deck profile)

| rate MB/s | lat ms | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 5 | 8 | 16 | 7690.5 (7689.5-7691.1) | 1573.5 (1572.3-1573.7) | **4.89x** | 100 | 99 | exact |
| 100 | 5 | 8 | 32 | 7693.4 (7692.7-7695.1) | 1575.4 (1575.0-1575.9) | **4.88x** | 100 | 99 | exact |
| 100 | 5 | 8 | 64 | 7700.5 (7699.6-7702.3) | 1582.5 (1581.6-1583.1) | **4.87x** | 100 | 99 | exact |

Anchor K = 8 matches the 10:55 / 05:55 / 02:55 cold rate-cap band (~4.87x–4.89x) within noise.

### random_read (rate 100 MB/s, lat 25 ms — scattered/seek penalty, cold DONTNEED)

| rate MB/s | lat ms | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 25 | 1 | 16 | 13917.4 (13904.6-13918.3) | 7787.8 (7774.5-7797.9) | **1.79x** | 55 | 20 | exact |
| 100 | 25 | 1 | 32 | 13919.9 (13905.8-13925.8) | 7797.2 (7787.2-7802.9) | **1.79x** | 55 | 20 | exact |
| 100 | 25 | 1 | 64 | 13926.9 (13923.3-13928.9) | 7808.4 (7798.7-7812.1) | **1.78x** | 55 | 20 | exact |
| 100 | 25 | 8 | 16 | 7710.1 (7709.8-7711.2) | 1593.8 (1593.0-1594.3) | **4.84x** | 100 | 98 | exact |
| 100 | 25 | 8 | 32 | 7713.3 (7713.2-7714.5) | 1596.5 (1596.0-1598.0) | **4.83x** | 100 | 98 | exact |
| 100 | 25 | 8 | 64 | 7720.0 (7719.5-7721.6) | 1603.0 (1602.0-1637.8) | **4.82x** | 99 | 97 | exact |

### mixed_rw (effective read 50 MB/s, lat 5 ms — write-contention half bandwidth, cold DONTNEED)

| rate MB/s | lat ms | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 5 | 1 | 16 | 16695.8 (16694.0-16704.6) | 4459.1 (4436.9-4464.9) | **3.74x** | 46 | 35 | exact |
| 50 | 5 | 1 | 32 | 16696.1 (16686.6-16698.2) | 4452.5 (4433.4-4461.0) | **3.75x** | 46 | 35 | exact |
| 50 | 5 | 1 | 64 | 16700.9 (16689.1-16708.9) | 4467.0 (4462.7-4491.6) | **3.74x** | 46 | 35 | exact |
| 50 | 5 | 8 | 16 | 15370.7 (15370.2-15370.9) | 3133.0 (3132.7-3133.3) | **4.91x** | 50 | 50 | exact |
| 50 | 5 | 8 | 32 | 15372.3 (15371.6-15373.3) | 3135.7 (3135.0-3135.8) | **4.90x** | 50 | 50 | exact |
| 50 | 5 | 8 | 64 | 15377.0 (15376.7-15377.8) | 3141.6 (3141.0-3142.9) | **4.89x** | 50 | 50 | exact |

### thermal (rate schedule 100→40→80 MB/s by progress thirds, lat 5 ms, K = 8, cold DONTNEED)

| rate schedule | lat ms | K | B | int8 ms | pair ms | pair vs int8 | int8 MB/s | pair MB/s | top-10 |
|---|---|---|---|---|---|---|---|---|---|
| 100/40/80 | 5 | 8 | 16 | 12150.3 (12150.0-12150.5) | 2479.4 (2479.2-2480.1) | **4.90x** | 63 | 63 | exact |
| 100/40/80 | 5 | 8 | 32 | 12153.1 (12152.2-12153.4) | 2482.4 (2482.2-2483.8) | **4.90x** | 63 | 63 | exact |
| 100/40/80 | 5 | 8 | 64 | 12158.7 (12158.5-12159.0) | 2489.4 (2488.9-2490.8) | **4.88x** | 63 | 63 | exact |

Harmonic-mean-ish sustained ~63 MB/s matches the 100/40/80 thirds schedule; pair stays in the ~4.88x–4.90x band.

## Verdict

**Viable** on both speed and memory. Under emulated random-read / mixed R/W / thermal-throttle Dev profiles (no physical media), **K = 8** keeps pair-vs-int8 in the **~4.82x–4.91x** band across random_read (100 MB/s @ 25 ms), mixed_rw (50 MB/s @ 5 ms), thermal (100→40→80), and the deck_anchor (100 MB/s @ 5 ms) at B ∈ {16, 32, 64}, N = 2M, exact top-10 — matching the prior Deck/SBC cold rate-cap band and well above the 1.5x speed bar. **K = 1** under random_read (25 ms seek) is latency-bound and drops to **~1.78x–1.79x** (still ≥1.5x viable on speed, but no longer in the ~4.8x band); mixed_rw K = 1 stays healthier at **~3.74x–3.75x**. Memory remains **4.92x smaller** (768,000,000 B vs 156,000,000 B). Answers the 10:55 Next question's software-Dev half: yes, the ~4.8x rate-cap lead holds under random seek, write-contention half-bandwidth, and mid-batch thermal throttle **when NCQ depth ≥ 8**; at K = 1 with heavy seek latency the lead compresses toward the 1.5x floor but stays above it. Bounded to this synthetic / Dev-model setup on a shared 8 vCPU box with no physical eMMC/SD and no writable cgroup `memory.high`.

Caveats: synthetic embeddings; modeled Dev rate+latency+NCQ / ThermalDev progress-fraction schedule (not device NCQ / physical media / kernel block layer / real SoC thermal governor); mixed_rw models write contention as a static half-rate rather than concurrent write IO; N = 2M; numpy / interpreter unpack overhead; shared 8 vCPU box; one seed, five repeats per cell.

## Pair files this wave

- No pair file for the 14:05 wave expected (Chief both pairs no gap for many idle waves; roadmap blobs still JuniorHome ROADMAP 4efb666 / JuniorLLM BETA_TO_OS bac6923). JuniorLLM tip before this push recorded from Chief last_tips was [e313737](https://github.com/cloudcover95/JuniorLLM/commit/e313737) (10:55 vault findings). The `ask` side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Emulated random-read / mixed R/W / thermal-throttle holds ~4.8x at K = 8 and ≥1.78x at K = 1 (seek-heavy). Remaining: **physical** eMMC/SD / real device NCQ when hardware exists; **or** a real writable cgroup `memory.high` (if delegation appears) that forces reclaim strictly below the int8 working set; **or** an emulated queue-depth sweep under combined random+thermal (K ∈ {1,2,4,8}) / asymmetric R/W ratios if physical media stays unavailable.

close: 2077021b3466dd78 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-06_1055)
