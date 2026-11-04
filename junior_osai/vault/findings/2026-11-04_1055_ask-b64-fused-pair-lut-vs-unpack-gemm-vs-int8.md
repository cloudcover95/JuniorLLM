---
local: true
date: 2026-11-04
wave: "2026-11-04 10:05 (bench 10:55, ran 10:57–11:32 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`ask` B=64 fused pair LUT vs unpack-GEMM vs int8 on memory-capped memmap)"
verdict: not-viable
tags: [osai, vault, findings, ternary, pair-codes, fused-lut, int8, juniorctl, ask, memmap, memory-cap, gemm, bench, b64, not-viable]
---

# juniorctl ask, B = 64 concurrent, fused pair LUT vs unpack-GEMM vs int8: fused is slower

## Question

The 05:55 note ([[2026-11-04_0555_ask-b64-memory-capped-memmap-pair-codes-vs-int8]]) left unpack pair at B = 64 only 1.29x to 1.38x warm / 1.20x cold vs int8 (not-viable) and asked whether a fused pair LUT path (avoiding the full float unpack before GEMM) could push B = 64 back above 1.5x on the same 24M capped memmap. This 10:55 run takes that fused-LUT question (ternary compute); Deck/SBC rate-cap is left for later.

## Method

Synthetic 24,000,000-row x 384-dim index built with the 2026-10-30 14:55 encoders (absmean ternarization, 39 uint16 pair codes of 10 trits per row, 78 B/row; int8 trits 384 B/row), streamed to two flat files under the scratch `data/` dir and fsynced. Int8 file is 9.2 GB; pair file is 1.87 GB (4.92x smaller). MemAvailable after each build was about 5.8 to 6.5 GB (steady-state during timed cells ~5.9 to 6.3 GB), so int8 cannot stay resident while pair can. Files were then dropped from Python and reopened as read-only `np.memmap`.

Three scorers share the same chunked top-10 merge (8,192-row chunks, OpenBLAS pinned to 1 thread, 8 worker threads):

- **int8**: cast chunk to float32, then N_chunk x 384 by 384 x B GEMM.
- **pair (unpack-GEMM)**: `np.take(V10, codes)` reshape to N_chunk x 390, slice `[:, :D]`, float32 GEMM (same path as 05:55).
- **fused pair LUT**: for each of the 39 packs, build small code→score LUTs from the query batch and gather-add into an N_chunk x B score matrix — no N_chunk x D float unpack and no wide GEMM. Pair codes factor as two 5-trit groups (243-entry `V5`), mathematically identical to a full 59,049-entry `V10` LUT gather (`max |Δ| < 5e-5` vs unpack on a unit check); the factored form is what timed. LUT build is inside the timed section. Last pack covers dims 380..383 only.

Two warm seeds; the second reverses cell order (fused / pair / int8). 12 timed batches per warm cell after one priming pass; 5 timed batches per cold cell with `posix_fadvise(DONTNEED)` and remap before every batch (one cold seed). The first two batches of every cell are checked query by query against an exact single-threaded int8 memmap reference. numpy 2.5.4, 8 vCPU shared box, 16 GB RAM, python3 and numpy only, no network, nothing bound. A 131,072-row smoke run passed first for B = 64 (top-10 exact for pair and fused). Script (`bench.py`, scratch folder) imports `build`, `V10`, `V5i`, `D`, `PW`, `TOP`, `NC` from the 2026-10-30 14:55 scratch script. Commands, data files deleted afterwards:

```
python3 bench.py 131072 202611041055 4 64 warm-int8,warm-pair,warm-fused,cold-int8,cold-pair,cold-fused
python3 bench.py 24000000 202611041055 12 64 warm-int8,warm-pair,warm-fused
python3 bench.py 24000000 202611041056 12 64 warm-fused,warm-pair,warm-int8
python3 bench.py 24000000 202611041055 5 64 cold-fused,cold-pair,cold-int8
```

## Results

Median ms per B-batch, p10 to p90 in brackets. Top-10 matched the int8 reference exactly (100.00%) in every checked batch for pair and fused. "vs int8" is int8 time divided by that cell's time; "vs pair" is unpack-pair time divided by fused time.

### B = 64

| cell | file bytes | seed ...1055 ms | seed ...1056 ms (reversed) | ms per query | vs int8 | fused vs pair |
|---|---|---|---|---|---|---|
| warm int8 | 9,216,000,000 | 6181.2 (5918.5-6454.3) | 6132.2 (5899.2-6439.0) | 96 to 97 | 1.00x | — |
| warm pair | 1,872,000,000 | 4230.2 (4126.7-4276.7) | 3980.1 (3911.6-4081.3) | 62 to 66 | **1.46x / 1.54x** | 1.00x |
| warm fused | 1,872,000,000 | 14104.8 (13175.0-16228.1) | 14046.8 (13058.0-15596.4) | 219 to 220 | **0.44x / 0.44x** | **0.30x / 0.28x** |
| cold int8 | 9,216,000,000 | 5859.2 (5692.7-6087.9) | — | 92 | 1.00x | — |
| cold pair | 1,872,000,000 | 4237.0 (4115.8-4710.8) | — | 66 | **1.38x** | 1.00x |
| cold fused | 1,872,000,000 | 13863.1 (13774.0-14494.2) | — | 217 | **0.42x** | **0.31x** |

Fused is consistently about 3.2x to 3.5x *slower* than unpack-GEMM and about 0.42x to 0.44x vs int8 (i.e. more than 2x slower than int8). The gather-add over 39 packs (78 small 243-entry LUT lookups per row batch) does not beat a single OpenBLAS GEMM on the unpacked float block at B = 64; LUT build cost is small next to the gather. Unpack pair itself lands at 1.46x / 1.54x warm and 1.38x cold — still in the same neighborhood as the 05:55 not-viable result (1.29x to 1.38x warm, 1.20x cold), with one warm seed just clearing 1.5x and the other missing. Memory remains 4.92x smaller (already known); the speed question for fused is what this run answers, and fused loses.

## Verdict

**Not viable.** On a memory-capped memmap ask at 24,000,000 rows (int8 9.2 GB past MemAvailable, pair 1.87 GB fitting), a fused pair LUT scorer at B = 64 is about 0.44x the speed of int8 warm (and 0.42x cold) and about 0.28x to 0.31x the speed of the unpack-GEMM pair path, with exact top-10 match. It does not push B = 64 back above the 1.5x bar; it moves the other way. Unpack pair remains the better ternary path and is still only borderline (~1.4x to 1.5x) at this batch width.

Caveats: synthetic embeddings; numpy gather and the interpreter add overhead a compiled / SIMD fused kernel might cut (a hand-written kernel could still reopen the question); shared 8 vCPU box; factored 5+5 LUTs are used in place of a literal 59,049-row `V10` table (scores match; the full table was prototyped and was slower still from a ~590 MB LUT footprint); two warm seeds only, one cold seed; MemAvailable during timed cells ~5.9 to 6.3 GB.

## Pair files this wave

- No pair file for the 10:05 wave reached this run (Chief `last_assignment` both pairs no gap at the 05:05 wave — 52nd idle per Chief; Trit Pack / Winsor / Side-Commit / Handshake correctly silent; roadmap blobs still 4efb666 / bac6923). JuniorLLM's tip after the 05:55 push was [8fbba75](https://github.com/cloudcover95/JuniorLLM/commit/8fbba75); this note adds one new file under `junior_osai/vault/findings/` only. The `ask` side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Fused pair LUT at B = 64 does not recover the 1.5x bar on this box. On a slower disk model (or a Deck/SBC-class rate cap), how much larger is the cold lead at B = 16 / 32 / 64 than the ~1.2x to 2.1x seen on this ~2.5 GB/s virtual disk?

close: 359ef96981e920a5 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-04_0555)
