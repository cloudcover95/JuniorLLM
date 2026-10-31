---
local: true
date: 2026-10-31
wave: "2026-10-31 02:05 (bench 02:55, ran 02:58 to 03:02)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl health` / `ask` / `night` / `security` + test_junior_aie + test_linux_security (`night` sweeping the packed ask index)"
verdict: not-viable
tags: [osai, vault, findings, ternary, pair-codes, int8, juniorctl, night, ask-index, chunk-size, working-memory, bench]
---

# juniorctl night: chunk-size sweep at 64 queries per chunk, uint16 pair-code decode vs int8 cast-and-matmul

## Question

The 18:55 note ([[2026-10-30_1855_night-fused-lut-scoring-pair-codes-vs-int8]]) found that batching all 64 queries per chunk matters far more than the storage format, and that at B = 64 the pair-table decode path was 0.82x to 0.86x of int8 at 8,192-row chunks. With B = 64 fixed, how do chunk sizes from 2,048 to 131,072 rows change int8 and decode sweep time and peak float32 working memory, and is there a chunk size where decode comes within 1.05x of int8?

## Method

Same synthetic index, encoding and `build()` as the 18:55 script (imported from it): 500,000 rows of 384 dims ternarized with the 0.5 mean|x| absmean rule, stored as int8 trits (384 B/row, 192,000,000 B) and as uint16 pair codes (39 codes of 10 trits, 78 B/row, 39,000,000 B), all in memory, no disk or device model, so this is compute only. All 64 queries per chunk, top-10, OpenBLAS pinned to 1 thread, a thread pool of 4 or 8 workers over chunks of 2,048, 4,096, 8,192, 16,384, 32,768 and 131,072 rows. Two paths per chunk: int8 rows cast to float32 then `Q @ X.T`, and the 59,049 x 10 int8 pair-table gather, cast, then `Q @ X.T`. Each cell ran 7 timed repeats with the variant order reversed on odd repeats, plus one untimed sweep under `tracemalloc` for the peak traced numpy allocation. Every sweep's top-10 was compared to an 8-thread, 8,192-row int8 reference. numpy 2.5.4, 8-vCPU shared box, MemAvailable about 6.5 GB at start, python3 and numpy only, no network, nothing bound. A 65,536-row smoke run (`python3 bench.py 65536 1 1 8 2048,131072`) came first. Command, run in a scratch folder (not committed):

```
python3 bench.py 500000 202610310255 7 4,8 2048,4096,8192,16384,32768,131072   # N, seed, repeats, threads, chunk sizes
```

Core of the script:

```
from bench import build, V10, D, TOP, recall          # the 2026-10-30 18:55 script
def sweep(kind,I8,PC,Q,T,C):
    def chunk(c):
        s=c*C
        if kind=="int8": X=I8[s:s+C].astype(np.float32)
        else: P=PC[s:s+C]; X=np.take(V10,P,axis=0).reshape(len(P),390)[:,:D].astype(np.float32)
        S=Q@X.T; j=np.argpartition(-S,TOP,1)[:,:TOP]; return np.take_along_axis(S,j,1),j+s
    with ThreadPoolExecutor(T) as ex: parts=list(ex.map(chunk,range(-(-len(I8)//C))))
    ...                                                  # merge per-chunk top-10s, argsort, take top-10
tracemalloc.start(); sweep(k,...); peak=tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
```

## Results

Median of 7 full 64-query sweeps over 500,000 rows in ms, spread (max minus min) in brackets. int8/decode above 1 means decode is faster. Peak MB is the traced numpy peak during one sweep (int8 / decode). Build took 8 s. Recall@10 against the reference was 1.0000 in every cell.

| threads | chunk rows | int8 ms | decode ms | int8/decode | peak MB int8 | peak MB decode |
|---|---|---|---|---|---|---|
| 4 | 2,048 | 95.0 (11.4) | 111.7 (8.8) | 0.85x | 21 | 23 |
| 4 | 4,096 | 88.2 (8.6) | 101.1 (5.2) | 0.87x | 40 | 43 |
| 4 | 8,192 | 81.7 (7.2) | 98.9 (7.9) | 0.83x | 78 | 79 |
| 4 | 16,384 | 77.7 (8.3) | 91.4 (4.5) | 0.85x | 168 | 168 |
| 4 | 32,768 | 122.3 (33.1) | 139.1 (12.1) | 0.88x | 336 | 294 |
| 4 | 131,072 | 149.1 (22.9) | 176.0 (61.2) | 0.85x | 1,280 | 1,045 |
| 8 | 2,048 | 52.8 (7.9) | 61.7 (9.2) | 0.86x | 40 | 38 |
| 8 | 4,096 | 42.7 (2.9) | 49.9 (4.6) | 0.86x | 78 | 75 |
| 8 | 8,192 | 46.5 (4.7) | 53.8 (5.7) | 0.87x | 168 | 154 |
| 8 | 16,384 | 53.2 (8.3) | 59.5 (4.0) | 0.90x | 331 | 298 |
| 8 | 32,768 | 70.4 (11.4) | 80.6 (14.0) | 0.87x | 654 | 604 |
| 8 | 131,072 | 153.4 (18.9) | 175.1 (116.5) | 0.88x | 1,079 | 1,007 |

A 3-repeat pass with the same seed (4 chunk sizes) gave the same picture: decode 0.73x to 0.91x of int8, best int8 at 8,192 rows and 8 threads (48.3 ms).

Decode never comes within 1.05x of int8: the best cell is 0.90x (16,384 rows, 8 threads) and the range is 0.83x to 0.90x. The gap is a nearly constant 12 to 17 percent at every chunk size, so it is the cost of the table gather itself, not a cache or scheduling effect that a chunk size can tune away.

Chunk size matters a lot for both paths, and in the same direction. At 8 threads the sweet spot is 4,096 rows (int8 42.7 ms, about 78 MB peak); 131,072 rows is 3.6x slower (153.4 ms) and uses about 14x the working memory (1,079 MB), because the float32 chunk (131,072 x 384 x 4 = 201 MB) blows every cache and only 4 chunks exist for 8 workers. At 4 threads the best is 16,384 rows (77.7 ms). Peak memory scales with chunk rows x workers in flight, and is essentially the same for both formats, since both expand to the same float32 matrix before the matmul.

## Verdict

**Not viable** on compute. At every chunk size from 2,048 to 131,072 rows and at 4 and 8 threads, the pair-code decode path is 0.83x to 0.90x of int8, with exact top-10 throughout, so no chunk size closes the gap to 1.05x. The pair format's win remains the bytes it stores and reads (78 vs 384 B/row, 4.92x smaller; 2.22x end to end at about 400 MB/s or slower in the 14:55 note), not compute. Practical setting for `night`: 4,096-row chunks at 8 workers (or 8,192 rows if fewer workers), which is up to 3.6x faster than very large chunks and keeps the float32 working set under about 80 MB.

Caveats: numpy `take` and `astype` stand in for what a native decode kernel would do; 500,000 rows in memory, synthetic embeddings, one seed, one shared 8-vCPU box, tracemalloc counts numpy allocations only (not OpenBLAS internal buffers).

## Pair files this wave

- No pair file for the 02:05 wave reached this run, and no new roadmap item is named; the roadmap blobs have had nothing uncovered since 2026-10-24, and the ledger will confirm the Chief's call. In JuniorLLM, the only commit in the last 6 hours is [40f749d](https://github.com/cloudcover95/JuniorLLM/commit/40f749d) (02:52 MT, cloudcover95, `rails/nopull/no_model_pull_check.py`, an offline no-model-pull / no-remote-script check). It is unattributed to a pair in this run, and this note does not touch `rails/`. This note adds one new file under `junior_osai/vault/findings/` only. The ask/night side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

With compute settled (int8 wins by 10 to 17 percent, pair codes win only on bytes), where is the crossover once the night sweep reads from storage at the best chunk size? At 4,096-row chunks, 8 workers and B = 64, at what sustained read rate (100 MB/s to uncapped) does reading 78 B/row pair codes and decoding beat reading 384 B/row int8, and how does that compare with the 8,192-row crossover from the 14:55 note?

close: f011323a216377a0 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-30_1855)
