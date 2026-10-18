---
local: true
date: 2026-10-18T02:55:00-06:00
wave: "2026-10-18 02:05"
roadmap_item: "JuniorOS overlay: Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH (hardened bitnetd.service part: MemoryMax=2G)"
roadmap_ref: "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay (Quant Chief pair_A item this wave, hardened bitnetd.service part only; Winsor 839a429)"
verdict: viable
tags: [osai, vault, findings, ternary, i2s, winsor, gamma, calibration, memorymax, bitnetd, streaming, benchmark]
---

# Ternary calibrate+pack under bitnetd MemoryMax=2G: streaming exact gamma vs full array

- Prev: [[2026-10-17_1855_live-beta-sprt-packed-fsm-vs-float]]
- Roadmap: JuniorOS overlay, hardened bitnetd.service part (JuniorLLM docs/BETA_TO_OS.md)
- Why this question: the Chief gave pair A the hardened `bitnetd.service` part. The unit (`rails/linux/bitnetd.service`) sets `MemoryMax=2G`. Winsor's file this wave (839a429) points out that the drop and p95 gammas need rank cuts of |W|, estimates a CPython-list sort at about 566 MB for one 2560x6912 projection (over a 1/16 calibration share of 128 MiB), and replaces it with a certified bracket from an O(B) streaming histogram. This note asks what the numpy path actually costs, whether numpy can make the streamed gamma exact instead of bracketed, and whether the whole calibrate-then-2-bit-pack step gets cheaper. The 18:55 note's next question (step-4 drift check on popcounts) is deferred so the bench follows the Chief's item.

## Question
For one BitNet-2B-class projection, can a chunked numpy pass compute the exact drop and p95 Winsor gammas and emit the 2-bit I2_S trit pack with chunk-bounded memory, bit-identical to the full-array path? Is it at least 1.5x smaller in peak memory or faster?

## Method
- Definitions match Winsor's `exact_gamma` and `quantize` in 839a429: drop = mean of the n - n//16 smallest |w|; p95 = mean of min(|w|, q) with q the ceil(0.95 n)-th smallest; trit is nonzero when |w|/gamma >= 0.5. Pack is the I2_S 2-bit code (0 to 00, +1 to 01, -1 to 10), four per byte, MSB first. The pack uses the drop gamma.
- Cross-check: on 131,072-weight arrays (Gaussian and heavy-tailed), both numpy paths gave gammas with 0.0 relative difference from Winsor's stdlib `exact_gamma` and packs byte-identical to Winsor's `pack_i2s(quantize(...))`.
- **full**: `a = np.abs(W)` (fp32 copy), one `np.partition` at both rank cuts, float64 sums, then full-array quantize and pack into an n/4 uint8 buffer.
- **stream**: chunks of C weights. Pass 1 takes max |w|. Pass 2 builds a 4,096-bin fixed-width histogram of counts and float64 sums with `np.bincount`. Pass 3 gathers only the values in the two bins that hold the rank cuts and partitions them, so both gammas are exact (no bracket slack). Pass 4 quantizes and packs each chunk into a preallocated n/4 buffer.
- Data: fp32, seed 18. "gauss" is N(0, 0.02). "t3_outrows" is Student-t with 3 degrees of freedom times 0.02, with 1/64 of rows scaled by 8 (outlier rows).
- Shapes: 2560x2560 (6.55M weights) and 2560x6912 (17.69M, the MLP projection size Winsor planned for).
- Peak memory is `tracemalloc` peak of numpy allocations during calibrate+pack, excluding W itself and including the n/4 output buffer. Time is 7 interleaved repeats, median (p25-p75), 1 process, 8-core box, numpy 2.5.4 (partition, bincount and the elementwise ops are single-threaded).
- Box-local results: `/workspace/tbench/r0255/results.jsonl` (not pushed).

```
cd /workspace/tbench/r0255
python3 memcap_gamma_pack_bench.py 7
```

Core of the script (full file at `/workspace/tbench/r0255/memcap_gamma_pack_bench.py`):

```python
def i2s_pack(w, g, out):                       # same function for both paths
    a = np.abs(w); a /= np.float32(g); nz = a >= 0.5; del a
    codes = nz.view(np.uint8) & (w > 0).view(np.uint8)
    codes |= (nz & (w < 0)).view(np.uint8) << 1
    c = codes.reshape(-1, 4)
    np.copyto(out, (c[:, 0] << 6) | (c[:, 1] << 4) | (c[:, 2] << 2) | c[:, 3])

# stream, pass 2: fixed-width bins are monotone in |w|, so each rank cut lands in one bin
idx = np.minimum((a * np.float32(B / mx)).astype(np.int32), B - 1)
cnt += np.bincount(idx, minlength=B); sm += np.bincount(idx, weights=a, minlength=B)
# pass 3: gather only the cut bins, partition them -> exact sum of the m smallest
v = np.partition(cut_bin_values, m - 1)
g_drop = (sum_below_cut_bin + v[:m].sum(dtype=np.float64)) / (n - n // 16)
```

## Results (C = 262,144 weights per chunk)

| shape | dist | full peak MB | stream peak MB | memory ratio | full ms | stream ms | speed ratio | cut-bin gather | pack bit diffs |
|---|---|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 34.41 | 6.39 | 5.38x | 69.0 (66.3-69.4) | 55.0 (52.3-56.8) | 1.26x | 2,318 | 0 |
| 2560x2560 | t3_outrows | 34.41 | 6.39 | 5.38x | 86.6 (82.6-88.2) | 55.4 (53.7-57.1) | 1.56x | 113,158 | 0 |
| 2560x6912 | gauss | 92.90 | 6.92 | 13.43x | 280.1 (276.5-293.8) | 144.9 (142.7-145.6) | 1.93x | 6,154 | 0 |
| 2560x6912 | t3_outrows | 92.90 | 9.66 | 9.62x | 250.5 (244.0-260.5) | 163.2 (160.2-169.8) | 1.54x | 489,642 | 0 |

Gamma differences between the two paths were 0.0 everywhere except 2560x6912 t3_outrows, where float64 summation order gave 7.8e-16 (drop) and 6.6e-16 (p95) relative, with 0 pack bit differences.

With C = 1,048,576 the stream peak rises to 25.27 MB: the memory ratio drops to 1.36x at 2560x2560 (below the bar) and 3.68x at 2560x6912, and the speed ratio is 1.17x to 1.70x. Smaller chunks win on both counts here.

Output and source sizes for scale (MB): 2-bit pack 1.64 and 4.42, fp16 13.11 and 35.39, fp32 26.21 and 70.78.

## Verdict
**Viable.** At C = 262,144 the streaming path computed the exact drop and p95 gammas and emitted a bit-identical 2-bit I2_S pack with 5.4x to 13.4x lower peak memory (6.4 to 9.7 MB vs 34 to 93 MB) and ran 1.26x to 1.93x faster (1.54x to 1.93x on the 2560x6912 projection). The memory bar clears on every shape and distribution, the speed bar on three of four.

- **Exact, not bracketed.** In numpy the third pass that gathers only the two cut bins (2,318 to 489,642 values, at most 2 MB as fp32) makes the streamed gammas exact, so Winsor's dSSE certificate isn't needed on this path: the trit stream equals the sort path by construction, and measured diffs were 0. Winsor's bracket still matters for the stdlib-only engines, which can't hold the cut bin cheaply.
- **Winsor's 566 MB is a CPython-list number.** The numpy full path peaks at 93 MB for 2560x6912, about 72% of the 128 MiB share, so it does fit for one projection. Streaming buys headroom (several projections calibrated at once, or a smaller share) and speed, not bare feasibility.
- **Where the speed comes from.** `np.partition` on 17.7M floats plus full-array temporaries is slower than four cache-sized chunk passes with `bincount`. This is a one-time calibrate/pack cost, not inference throughput.
- **Ternary-specific part.** Once gammas are known, the 2-bit pack is the only thing that has to be resident (n/4 bytes, 4.4 MB here vs 35.4 MB for fp16), so the stream peak is mostly the output buffer plus about 2.5 MB of chunk temporaries.
- **Caveat.** The gather size depends on the distribution: fixed-width bins put more tail values in one cut bin for heavy tails (489,642 vs 6,154). A pathological tensor could put most weights in one bin; a guard that re-bins the cut bin when it exceeds a budget would bound that.

Suggested wiring, for the pair to decide: Winsor's file can report the numpy exact-stream gamma next to its bracket when numpy is present, and keep the bracket as the stdlib fallback; Trit Pack's key can then take the pack bytes from pass 4 directly.

## Ops check
In local `/workspace/JuniorLLM` (offline, blender off), `python3 scripts/terraform_lean_prod.py code` and `... llm` both exited 0 with `"ok": true`, `"bind": "127.0.0.1"`, `"bpy": false`, `"model_pull": false` and `"launch": false`.

## This wave's pair files (2026-10-18 02:05)
- Chief assignment: pair_A JuniorOS overlay, hardened bitnetd.service part only; pair_B Live beta exit-code side + test_linux_security.
- Winsor & Quant Error: [JuniorStock quant/calibration/bitnetd_memcap_stream_gamma.py @ 839a429](https://github.com/cloudcover95/JuniorStock/blob/839a429/quant/calibration/bitnetd_memcap_stream_gamma.py) (JuniorOS overlay, hardened bitnetd.service, quant-error side)
- Trit Pack & I2_S: no 02:20 push found in local copies at 02:55.
- Side-Commit & Rate Lock: [web3node linux_security_finding_ledger.py @ 344d43d](https://github.com/cloudcover95/web3node/blob/344d43d/linux_security_finding_ledger.py) (BETA_TO_OS Live beta exit-code side)
- Handshake & Ports: [JuniorLLM ports/live_beta_exit_code_port_map.py @ 062bfc7](https://github.com/cloudcover95/JuniorLLM/blob/062bfc7/ports/live_beta_exit_code_port_map.py) (BETA_TO_OS Live beta exit-code contract)

## Next question
The hardened unit's MemoryMax counts page cache against the cgroup. When the source weights are an mmap'd fp16/bf16 file read chunk by chunk, does the streaming calibrate+pack still stay bounded (RSS plus touched pages), and does dropping pages per chunk cost more than the 1.5x it saves? Or pick up the 18:55 note's step-4 drift-on-popcounts question if the Chief assigns Live beta again.

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-17_1855_live-beta-sprt-packed-fsm-vs-float.md.

close: e21c48e171a2601c
