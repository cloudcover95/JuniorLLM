---
local: true
date: 2026-10-18T05:55:00-06:00
wave: "2026-10-18 05:05"
roadmap_item: "Live beta: juniorctl health / ask / night + tests/test_junior_aie.py (juniorctl ask retrieval + test_ask_and_cache part)"
roadmap_ref: "JuniorLLM docs/BETA_TO_OS.md Live beta steps 1-3 (Quant Chief pair_A item this wave; Winsor JuniorQuant 946f352, Trit Pack JuniorLLM 184a480)"
verdict: viable
tags: [osai, vault, findings, ternary, i2s, winsor, retrieval, juniorctl, ask, cache, memory, benchmark]
---

# juniorctl ask retrieval: exact top-1 from a resident 2-bit ternary index vs fp32, fp16 and int8

- Prev: [[2026-10-18_0255_memcap-stream-exact-gamma-pack-vs-full]]
- Roadmap: Live beta steps 1-3, `juniorctl ask` and `test_ask_and_cache` (JuniorLLM docs/BETA_TO_OS.md)
- Why this question: the Chief gave pair A the Live beta health/ask/night + test_junior_aie item. Winsor's file this wave (946f352) stores the document side of the ask index ternary and adds a per-query certificate: split each row as d = alpha * d_hat + r with r orthogonal to d_hat, keep alpha and rho = ||r||, and call a cached ternary answer sound only when the winner's lower bound beats every other row's upper bound. Running Winsor's own synthetic (`--synthetic 200 --queries 300 --seed 7`) certifies only 2% of queries under absmean and p95 and 0% under drop, so certify-or-refuse would refuse almost every ask. This note asks whether the same alpha/rho bound can instead drive an exact rerank, and what a resident 2-bit index then costs in memory and latency against the float baselines.

## Question
For B=1 `ask` top-1 over 20k to 100k document embeddings, can a resident 2-bit ternary index plus Winsor's alpha/rho bound return exactly the fp32 answer by rescoring only the rows the bound can't rule out, and is it at least 1.5x smaller or faster than a resident fp32 (or fp16, int8) index?

## Method
- Ternary code per row as in Winsor's `build_index`: absmean gamma, trit nonzero when |d|/gamma >= 0.5; alpha (least-squares scale) and rho computed in float64 and stored as fp32. Index is the I2_S-style 2-bit code (0 to 00, +1 to 01, -1 to 10, four per byte, MSB first), checked to decode back to the exact trits.
- Ternary scores: a byte LUT per query, `L[j, b] = <DEC[b], q[4j:4j+4]>` (d/4 x 256 fp32), then `sum_j L[j, P[:, j]]` over 8,192-row chunks. A decode-to-fp32-then-BLAS variant was 3.4-3.7x slower and is shown for reference only.
- Exact rerank (new here, not in 946f352): rescore the top 8 ternary rows from the fp32 rows on disk (`np.memmap`), take s0 = best exact score, then rescore every row with `alpha * s_hat + ||q|| * rho + eps >= s0`. Any row below that bound has float score < s0, so the returned row is the fp32 top-1 by construction. eps = 64 * d * 2^-24 * ||q|| * (|alpha| * ||d_hat|| + ||d|| + rho), a generous fp32 rounding allowance.
- Same bound with a per-row absmax int8 index as the stronger compressed baseline. fp16 is chunked upcast plus BLAS (numpy has no fp16 BLAS). fp32 is one `D @ q`.
- Data: "dense" = 256 Gaussian centers plus 0.8 per-dim noise, L2-normalized, at 100k x 384 (seed 55) and 50k x 768 (seed 56); queries are a random doc plus Gaussian noise of norm 0.5 ("near") or 1.2 ("far"), 200 each. "winsor_bow" = Winsor's own hashed bag-of-words `embed` and `synthetic` generator (loaded read-only from the 946f352 file) at 20k docs x 256 dims, 200 queries.
- Timing: per-query wall time, median over 200 queries, 5 repeats, reported as median (min-max of the per-repeat medians). One BLAS thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`), 8-core box, numpy 2.5.4, nothing bound, no network.
- Resident bytes: fp32 4nd, fp16 2nd, int8 nd plus 3 floats per row, ternary nd/4 plus 2 floats per row (alpha*gamma and rho). The fp32 file for reranks stays on disk; the bytes it touches per query are listed separately.
- Box-local results: `/workspace/tbench/r0555/results.jsonl` (not pushed).

```
cd /workspace/tbench/r0555
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 ask_index_bench.py 5
```

Core of the script (full file at `/workspace/tbench/r0555/ask_index_bench.py`):

```python
def scores_t2_lut(P, q, off):                      # P: n x d/4 uint8, 2-bit trits
    L = (q.reshape(m, 4) @ DEC.T).ravel()          # L[j*256+b] = <DEC[b], q[4j:4j+4]>
    for i in range(0, n, CH):
        idx = P[i:i+CH].astype(np.int32); idx += off
        out[i:i+CH] = L[idx].sum(1)

def exact_from_index(shat_scaled, ub_extra, mm, q, k0=8):   # shat_scaled = alpha*gamma*<t,q>
    top = np.argpartition(-shat_scaled, k0)[:k0]
    s0 = (np.asarray(mm[np.sort(top)]) @ q).max()            # exact scores from fp32 file
    cand = np.union1d(np.flatnonzero(shat_scaled + ub_extra >= s0), top)  # ub_extra = ||q||*rho + eps
    s = np.asarray(mm[cand]) @ q
    return cand[np.argmax(s)], s.max(), cand.size
```

## Results

Latency per query, ms, median (min-max over 5 repeats):

| set | queries | fp32 full | fp16 full | int8 + bound | ternary scores only | ternary + bound (exact) |
|---|---|---|---|---|---|---|
| dense 100k x 384 | near | 5.11 (5.07-5.15) | 46.85 (45.09-48.00) | 11.75 (11.54-11.95) | 29.20 (28.93-29.96) | 29.55 (29.30-30.46) |
| dense 100k x 384 | far | 4.96 (4.92-5.28) | 46.74 (46.26-47.49) | 12.04 (11.39-12.27) | 28.93 (28.84-29.45) | 29.46 (29.30-29.85) |
| dense 50k x 768 | near | 4.85 (4.77-4.87) | 46.00 (43.77-46.68) | 11.40 (11.00-11.54) | 28.73 (28.47-28.87) | 29.01 (28.74-29.60) |
| dense 50k x 768 | far | 4.91 (4.86-5.02) | 46.58 (45.81-47.49) | 11.91 (11.63-12.38) | 28.50 (27.68-28.54) | 29.07 (29.03-29.47) |
| winsor_bow 20k x 256 | Winsor synthetic | 0.66 (0.64-0.66) | 12.31 (12.11-12.42) | 1.71 (1.71-1.78) | 3.86 (3.78-3.90) | 4.06 (4.01-4.08) |

Memory, exactness and rerank work:

| set | queries | resident MB fp32 / fp16 / int8 / ternary | fp32 / ternary | int8 / ternary | ternary rows rescored, median / p95 / max | rerank bytes touched (median) | int8 rows rescored | top-1 diffs vs fp32 (ternary, int8) | raw ternary top-1 agrees | Winsor certified |
|---|---|---|---|---|---|---|---|---|---|---|
| dense 100k x 384 | near | 153.6 / 76.8 / 39.6 / 10.4 | 14.77x | 3.81x | 198.5 / 334 / 374 | 0.30 MB | 8 | 0 / 0 of 200 | 100% | 0% |
| dense 100k x 384 | far | 153.6 / 76.8 / 39.6 / 10.4 | 14.77x | 3.81x | 408 / 715 / 2,154 | 0.63 MB | 8 | 0 / 0 of 200 | 100% | 0% |
| dense 50k x 768 | near | 153.6 / 76.8 / 39.0 / 10.0 | 15.36x | 3.90x | 110.5 / 170 / 200 | 0.34 MB | 8 | 0 / 0 of 200 | 100% | 0% |
| dense 50k x 768 | far | 153.6 / 76.8 / 39.0 / 10.0 | 15.36x | 3.90x | 198 / 221 / 243 | 0.61 MB | 8 | 0 / 0 of 200 | 100% | 0% |
| winsor_bow 20k x 256 | Winsor synthetic | 20.48 / 10.24 / 5.36 / 1.44 | 14.22x | 3.72x | 30 / 155 / 225 | 0.03 MB | 8 | 0 / 0 of 200 | 67% | 0% |

Median rho / ||d|| was 0.46 for ternary on the dense sets and 0.22 on winsor_bow, against 0.007 and 0.003 for int8. That is why int8 never needs more than its 8 seed rows and ternary needs 30 to 2,154. Top-1 scores matched fp32 within 1e-6 on all 1,000 queries for both bounded paths. The byte LUT matched the decode path within 1e-4.

## Verdict
**Viable on memory, not on speed.** The resident 2-bit index plus Winsor's alpha/rho bound returned exactly the fp32 top-1 on all 1,000 queries while holding 14.2x to 15.4x less in memory than fp32 (10.4 MB vs 153.6 MB for 100k x 384), 7.1x to 7.7x less than fp16 and 3.7x to 3.9x less than int8. The cost is latency: about 29 ms per ask at 100k docs, 5.8x to 6.2x slower than one-thread fp32 BLAS and 2.4x to 2.5x slower than the int8 index with the same bound, though 1.6x to 3.0x faster than numpy fp16.

- **Certify-or-refuse doesn't work at this rho; certify-then-rerank does.** Winsor's certificate passed 0% of queries here, because ||q|| * rho (about 0.46 for dense ternary rows) swamps typical score gaps. Using the same bound to pick rows to rescore needs a median of 0.15% to 0.41% of rows (worst case 2.2%) and makes the answer exact, so a cache entry can always store the fp32 answer.
- **Caching raw ternary answers would be wrong.** On Winsor's own bag-of-words data the raw ternary top-1 matched fp32 on only 67% of queries. The rerank fixed every one of them for a median of 30 extra rows.
- **When to pick which.** If memory is the constraint (the hardened `bitnetd.service` at MemoryMax=2G, the Deck, or many indexes at once), ternary plus the bound gives exact answers in about a quarter of the int8 footprint at about 30 ms per ask. If int8 fits, int8 plus the bound is the better trade: exact, 3.8x to 3.9x smaller than fp32, and only 2.3x to 2.6x slower.
- **Caveat.** The rerank reads fp32 rows from a file, so that file must be on disk (0.03 to 0.63 MB touched per query here, plus the page cache it pulls in). The far-query worst case (2,154 rows) suggests clustered or near-duplicate corpora can grow the rescored set; a cap with a fall back to a full fp32 scan would bound latency.

Suggested wiring, for the pair to decide: Winsor's file could report the rescored-set size next to its certified fraction and treat "exact after rerank" as cache-sound; Trit Pack's contract key could carry one trit for "ask answer came from the exact path".

## Ops check
In local `/workspace/JuniorLLM` (offline, blender off), `python3 scripts/terraform_lean_prod.py code` and `... llm` both exited 0 with `"ok": true`, `"bind": "127.0.0.1"`, `"bpy": false`, `"model_pull": false` and `"launch": false`.

## This wave's pair files (2026-10-18 05:05)
- Chief assignment: pair_A Live beta health/ask/night + test_junior_aie; pair_B JuniorOS overlay, juniorctl on PATH + Debian/Alpine part.
- Winsor & Quant Error: [JuniorQuant src/live_beta_ask_cache_rank_cert.py @ 946f352](https://github.com/cloudcover95/JuniorQuant/blob/946f352/src/live_beta_ask_cache_rank_cert.py) (Live beta ask cache, quant-error side)
- Trit Pack & I2_S: [JuniorLLM scripts/live_beta_aie_ctl_contract_key.py @ 184a480](https://github.com/cloudcover95/JuniorLLM/blob/184a480/scripts/live_beta_aie_ctl_contract_key.py) (Live beta test_junior_aie + juniorctl contract key)
- Side-Commit & Rate Lock: [stocksnode src/stocksnode/core/juniorctl_path_base_receipt_ledger.py @ c2513f4](https://github.com/cloudcover95/stocksnode/blob/c2513f4/src/stocksnode/core/juniorctl_path_base_receipt_ledger.py) (JuniorOS overlay, juniorctl on PATH + Debian/Alpine receipts)
- Handshake & Ports: [JuniorHome ports/overlay_netinst_path_port_map.py @ ac08e31](https://github.com/cloudcover95/JuniorHome/blob/ac08e31/ports/overlay_netinst_path_port_map.py) (JuniorOS overlay netinst PATH port map)

## Next question
Can a wider lookup make the ternary scan cheaper: a 16-bit (two-byte) LUT halves the gathers but grows the table to d/8 x 65,536 entries, so does it beat the 8-bit LUT's ~29 ms at 100k docs, or does cache pressure eat the gain? Or, if the Chief moves on, take the 02:55 note's mmap page-cache question under MemoryMax.

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-18_0255_memcap-stream-exact-gamma-pack-vs-full.md.

close: 4540bcd5c43a0775
