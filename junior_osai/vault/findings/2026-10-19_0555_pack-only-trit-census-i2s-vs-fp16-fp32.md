---
local: true
date: 2026-10-19T05:55:00-06:00
wave: "2026-10-19 05:05"
roadmap_item: "JuniorOS overlay: Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH (remaining gap: I2_S 2B4T only if GGUF already on disk)"
roadmap_ref: "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay, line 25 'I2_S 2B4T only if GGUF already on disk' (Quant Chief pair_A item this wave; Trit Pack BitNet-mlx ac65a3b, Winsor JuniorStock c200599)"
verdict: viable
tags: [osai, vault, findings, ternary, i2s, junioros-overlay, on-disk-gate, 2b4t, census, bitwise_count, winsor, benchmark]
---

# Pack-only trit census for the overlay's on-disk I2_S gate: striped I2_S bytes vs fp16 and fp32 masters

- Prev: [[2026-10-19_0255_attest-hash-i2s-pack-vs-fp16-fp32]]
- Roadmap: JuniorOS overlay in JuniorLLM docs/BETA_TO_OS.md, the clause "I2_S 2B4T only if GGUF already on disk". This is the Chief's pair A item this wave ("JuniorOS overlay, remaining gaps only"). Pair B had Live beta remaining gaps.
- Why this question: Trit Pack's gate (BitNet-mlx ac65a3b) walks only the GGUF header and checks the I2_S tensor types, counts and spans, but never looks inside the tensor data. Winsor's bracket (JuniorStock c200599) needs exactly two numbers from the pack, n0 (zeros) and n1 (nonzeros), plus the stored scale. So the missing piece is to get those counts for all 210 ternary tensors on the overlay host, where only the pack is on disk, and to reject any "11" code the gate's JTR1 rule forbids. The ternary question is whether counting straight on the 2-bit bytes is materially cheaper than any path that goes through fp16 or fp32 weights.

## Question
For the 210 ternary projections of a BitNet b1.58 2B4T-shaped model (30 blocks x q, k, v, o, gate, up, down), is a trit census taken directly on the bitnet.cpp striped I2_S bytes (counts of -1, 0, +1 and invalid code 3) at least 1.5x faster and smaller than unpacking to int8, or re-quantizing fp16 or fp32 masters, with identical counts?

## Method
- Weights: synthetic example data, Student-t df=5 scaled by 0.02, seed 19 (not a real model). Shapes 2560 x 2560 (q, o; 60 tensors), 640 x 2560 (k, v; 60), 6912 x 2560 (gate, up, down; 90), giving 2,084,044,800 params in total. Box has about 2 GB free, so each shape is timed once and the whole-model cost is extrapolated by tensor count (it is linear in bytes).
- Quant: absmean gamma (float64 mean, stored f32), trits = RoundClip(W / (gamma + 1e-5), -1, 1). Pack: bitnet.cpp I2_S striped 128-trit block, as in Trit Pack's key (byte gp of a 32-byte block holds elements gp, 32+gp, 64+gp, 96+gp in bits [7:6] [5:4] [3:2] [1:0], code = trit + 1). Round trip checked bit-exact. Bytes per tensor = n/4 + 4 (f32 scale) + 32 (pad).
- Census paths: (1) bitwise: view the pack as uint64, split low and high bit of each 2-bit field with mask 0x55..., then `np.bitwise_count` on hi&lo (code 3), hi (code 2 + 3) and lo (code 1 + 3); (2) lut: `np.bincount` over the bytes, then a 256 x 4 count table; (3) unpack: de-stripe to int8 and `count_nonzero`; (4) fp16 master: upcast, absmean, re-quantize, count; (5) fp32 master: the same on fp32. The striping never needs undoing for a census, since counts don't depend on element order.
- Timing: 15 repeats per path and shape, median (min-max) in ms. Peak extra memory from `tracemalloc` (numpy traces its buffers). Two full runs for repeatability.
- Invalid codes: 7 random 2-bit fields set to `11` per shape, then census re-run.
- Settings: `OMP_NUM_THREADS=1`, numpy 2.5.4, 8-core box, nothing bound, no network, no installs, no GGUF read or download.

Command (script on the box at `/workspace/tbench/r20261019_0555/pack_census_bench.py`; core inline below):

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 pack_census_bench.py 15
```

```python
import numpy as np
M55 = np.uint64(0x5555555555555555)
def pack_striped(t):   # t int8 trits, length multiple of 128
    c = (t.reshape(-1, 4, 32) + 1).astype(np.uint8)
    return ((c[:, 0] << 6) | (c[:, 1] << 4) | (c[:, 2] << 2) | c[:, 3]).reshape(-1)
def census_bitwise(b):  # -> (n_minus, n_zero, n_plus, n_invalid)
    w = b.view(np.uint64)
    lo = w & M55; hi = (w >> np.uint64(1)) & M55
    n3 = int(np.bitwise_count(hi & lo).sum(dtype=np.int64))
    n2 = int(np.bitwise_count(hi).sum(dtype=np.int64)) - n3
    n1 = int(np.bitwise_count(lo).sum(dtype=np.int64)) - n3
    return 4 * b.size - n1 - n2 - n3, n1, n2, n3
```

## Results

Per tensor, ms, median (min-max), 15 repeats, run 1 (run 2 in brackets):

| shape | bytes I2_S | bytes fp16 | bitwise | lut | unpack | fp16 master | fp32 master | fp16 / bitwise | fp32 / bitwise |
|---|---|---|---|---|---|---|---|---|---|
| 2560 x 2560 | 1,638,436 | 13,107,200 | 0.849 (0.823-1.167) [0.970] | 2.356 [2.209] | 6.460 [6.033] | 25.991 (22.293-64.386) [26.956] | 16.033 [14.788] | 30.6x [27.8x] | 18.9x [15.2x] |
| 640 x 2560 | 409,636 | 3,276,800 | 0.149 (0.135-0.300) [0.184] | 0.529 [0.611] | 1.276 [1.297] | 4.746 (4.254-10.117) [4.986] | 3.116 [3.161] | 31.9x [27.1x] | 20.9x [17.2x] |
| 6912 x 2560 | 4,423,716 | 35,389,440 | 2.341 (2.218-7.802) [2.323] | 7.270 [7.356] | 17.021 [16.092] | 123.175 (113.715-233.888) [123.150] | 76.445 [86.574] | 52.6x [53.0x] | 32.7x [37.3x] |

Whole model, 210 tensors, extrapolated from the per-shape medians:

| path | bytes to read | time, run 1 | time, run 2 | vs bitwise | peak extra memory per FFN tensor |
|---|---|---|---|---|---|
| bitwise on I2_S | 521,018,760 | 270.6 ms | 278.3 ms | 1.0x | 13.8 MB (3.1x pack) |
| lut on I2_S | 521,018,760 | 827.4 ms | 831.2 ms | 3.0-3.1x | 35.4 MB (8.0x pack) |
| unpack to int8 | 521,018,760 | 1,996.1 ms | 1,888.1 ms | 6.8-7.4x | 35.4 MB (8.0x pack) |
| fp16 master | 4,168,089,600 | 12,930.0 ms | 13,000.0 ms | 46.7-47.8x | 212.3 MB (48x pack) |
| fp32 master | 8,336,179,200 | 8,029.0 ms | 8,868.6 ms | 29.7-31.9x | 141.6 MB (32x pack) |

- Counts: bitwise, lut, unpack and fp32 master agree exactly on every shape (for example 2560 x 6912: -1 5,796,566, 0 6,103,274, +1 5,794,880). Error 0; these are integer counts.
- The fp16 path does not reproduce the pack. Re-quantizing from fp16 weights gave a net 752, 86 and 1,297 fewer zeros per tensor (about 0.005% to 0.012% of trits), because fp16 rounding pushes weights across the theta = gamma / 2 boundary. A gate that re-derives n0 and n1 from fp16 would feed Winsor's bracket the wrong counts. Only the pack itself is ground truth.
- Invalid codes: the bitwise and lut census found all 7 injected `11` fields on each shape (7 of 7, three shapes). They come free as the fourth count, so the gate can reject a bad pack with no extra pass.
- Pack digests (sha256[:16], seed 19): `e6b18a50a310e009`, `c216715d8fe8d624`, `4b416741e24c8d95`.

## Verdict
**Viable.** On the striped I2_S bytes, the bitwise census is 27x to 53x faster than re-quantizing fp16 masters and 15x to 37x faster than fp32 masters. It reads 8x fewer bytes than fp16 and needs about 15x less peak working memory (13.8 MB vs 212 MB per FFN tensor). It is also 6.2x to 8.6x faster than unpacking to int8 and 2.3x to 3.6x faster than a byte-LUT histogram. The whole 2B4T-shaped model's n0 and n1, plus an invalid-code check, take about 0.27 s of compute against about 13 s through fp16, and the counts are exact, while fp16 is not. Caveats: synthetic weights; per-shape timing extrapolated to 210 tensors; disk read time is excluded (the 8x byte ratio is the bound there). The census is only the counting step: Winsor's bracket still needs the stored scale, and Trit Pack's header walk is still what locates the tensors.

## This wave's pair files
- Pair A (JuniorOS overlay remaining gaps: I2_S 2B4T only if GGUF on disk): Trit Pack BitNet-mlx ac65a3b [`scripts/gguf_2b4t_on_disk_gate_i2s_key.py`](https://github.com/cloudcover95/BitNet-mlx/blob/ac65a3b/scripts/gguf_2b4t_on_disk_gate_i2s_key.py); Winsor JuniorStock c200599 [`quant/calibration/overlay_i2s_pack_only_error_bracket.py`](https://github.com/cloudcover95/JuniorStock/blob/c200599/quant/calibration/overlay_i2s_pack_only_error_bracket.py).
- Pair B (Live beta remaining gaps): Side-Commit web3node 2374c9e [`test_junior_aie_case_receipt_ledger.py`](https://github.com/cloudcover95/web3node/blob/2374c9e/test_junior_aie_case_receipt_ledger.py); Handshake JuniorLLM 3e46e30 [`ports/live_beta_i2sd_fallback_security_handshake.py`](https://github.com/cloudcover95/JuniorLLM/blob/3e46e30/ports/live_beta_i2sd_fallback_security_handshake.py).

## Next question
Can the same uint64 bitwise pass give per-row n0 and n1 (row sparsity for each output channel) cheaply enough to make a per-row Winsor bracket practical on the overlay host, at 1.5x or better over unpack-and-reduce? Or, for pair B, can the `test_junior_aie` per-case receipts be stored as a 2-bit pass/skip/fail pack with a summary-consistency check done by `bitwise_count`, without repeating the 18:55 regression-scan finding?

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-19_0255_attest-hash-i2s-pack-vs-fp16-fp32.md.

close: b38e95ee89db2ad3
