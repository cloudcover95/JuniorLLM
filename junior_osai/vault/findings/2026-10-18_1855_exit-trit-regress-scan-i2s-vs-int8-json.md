---
local: true
date: 2026-10-18T18:55:00-06:00
wave: "2026-10-18 18:05"
roadmap_item: "Live beta: end-to-end exit-code trits + test_linux_security (backfill_check cross-run regression and summary-consistency side)"
roadmap_ref: "JuniorLLM docs/BETA_TO_OS.md Live beta (Quant Chief pair_B item this wave; Side-Commit web3node c4dd23c, Handshake JuniorLLM fbfe7a1)"
verdict: viable
tags: [osai, vault, findings, ternary, i2s, live-beta, exit-trits, test_linux_security, backfill_check, popcount, benchmark]
---

# Exit-trit history: regression scan and summary counts on I2_S-packed bytes vs int8 trits and JSON-lines receipts

- Prev: [[2026-10-18_1455_append-only-t2-delta-vs-fp16-int8-8gb]]
- Roadmap: Live beta in JuniorLLM docs/BETA_TO_OS.md, end-to-end exit-code trits plus `tests/test_linux_security.py`, the Chief's pair B item this wave (pair A had JuniorOS overlay os-release.junior gaps).
- Why this question: Side-Commit's ledger (web3node c4dd23c, `live_beta_e2e_exit_trit_ledger.py`) maps every exit code to a trit (+1 pass, 0 not exercised, -1 fail), packs them I2_S style (0->0, +1->1, -1->2, MSB first) and asks backfill_check to list cross-run regressions (+1 in run r-1, -1 in run r). Handshake's port check (JuniorLLM fbfe7a1, `ports/live_beta_security_exit_trit_handshake.py`) uses the same 2-bit pack. Both store the pack only as a close-hash input. The ternary question is whether the packed bytes are also the faster form to query, so backfill never has to unpack or re-parse receipts.

## Question
On a long Live beta history (many runs x many steps or per-test rows), is a regression scan and a per-run pass/skip/fail count done directly on the I2_S 2-bit packed bytes at least 1.5x faster or smaller than the same scan on int8 trit arrays and on JSON-lines receipts, with identical results?

## Method
- History: synthetic per-step state machine, seeds 11, 23, 37. Start 95% pass / 5% skip; each run a pass flips to fail with p 0.002, a fail recovers with p 0.3, a pass skips with p 0.005, a skip returns with p 0.2. Shapes 4096 runs x 256 steps, 16384 x 1024, 4096 x 8192.
- Packed trick, no unpack: in an I2_S byte the low bit of each 2-bit field is the "+1" bit (mask 0x55) and the high bit is the "-1" bit (mask 0xAA). Regression bits = `((prev & 0x55..) << 1) & cur & 0xAA..` on uint64 views (32 trits per word), counted with `np.bitwise_count`. Per-run counts: pass = popcount(w & 0x55..), fail = popcount(w & 0xAA..), skip = steps - pass - fail.
- Baselines: int8 trits with `(t[:-1]==1) & (t[1:]==-1)` and `count_nonzero`; JSON lines `{"run": r, "trits": [...]}` parsed with `json.loads` and list counting (smallest shape only).
- Error check: regression counts and per-run counts asserted equal across all three forms on every shape (JSON on the smallest).
- Timing: 15 repeats (pack cost 5, JSON 3), median (min-max), `OMP_NUM_THREADS=1`, numpy 2.5.4, 8-core box, nothing bound, no network, no installs.

Command (script kept on the box at `/workspace/tbench/r20261018_1855/exit_trit_regress_bench.py`; core inline below):

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 exit_trit_regress_bench.py 15
```

```python
import numpy as np
M55, MAA = np.uint64(0x5555555555555555), np.uint64(0xAAAAAAAAAAAAAAAA)
def pack_i2s(t):  # t: int8 (runs, steps), steps % 32 == 0
    c = np.where(t == -1, 2, t).astype(np.uint8).reshape(t.shape[0], -1, 4)
    return (c[..., 0] << 6) | (c[..., 1] << 4) | (c[..., 2] << 2) | c[..., 3]
def packed_regress(p):
    w = p.view(np.uint64)
    return np.bitwise_count(((w[:-1] & M55) << np.uint64(1)) & w[1:] & MAA).sum(1)
def packed_counts(p, steps):
    w = p.view(np.uint64)
    pos = np.bitwise_count(w & M55).sum(1); neg = np.bitwise_count(w & MAA).sum(1)
    return np.stack([pos, steps - pos - neg, neg], 1)
def int8_regress(t): return np.count_nonzero((t[:-1] == 1) & (t[1:] == -1), axis=1)
```

## Results

| runs x steps | regressions | int8 regress ms | I2_S regress ms | speedup | int8 counts ms | I2_S counts ms | speedup | bytes int8 | bytes I2_S | JSON bytes / scan ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 x 256 | 2,061 | 0.671 (0.630-0.941) | 0.125 (0.119-0.209) | 5.4x | 1.615 (1.491-1.691) | 0.176 (0.173-0.295) | 9.2x | 1,048,576 | 262,144 | 3,253,732 / 91.9 (91.1-93.3) |
| 16384 x 1024 | 32,278 | 12.08 (10.87-13.43) | 1.618 (1.553-2.656) | 7.5x | 24.34 (21.40-25.55) | 2.097 (1.953-2.580) | 11.6x | 16,777,216 | 4,194,304 | not run |
| 4096 x 8192 | 64,824 | 24.47 (20.95-26.04) | 2.722 (2.428-3.994) | 9.0x | 50.40 (47.56-52.33) | 3.198 (3.083-5.432) | 15.8x | 33,554,432 | 8,388,608 | not run |

- Error: 0. Regression and count vectors are identical across I2_S, int8 and JSON on every shape.
- Memory: I2_S is 4.0x smaller than int8 and 12.4x smaller than JSON lines.
- Regress + counts together at 4096 x 256: I2_S 0.30 ms, int8 2.29 ms (7.6x), JSON 91.9 ms (about 305x).
- Pack cost (int8 to I2_S, once per write): 1.26 / 26.2 / 54.9 ms. It is paid at write time anyway, since the ledgers already compute the pack for the close hash. If a reader had only int8 and packed just to scan once, the win disappears at the smallest shape (1.26 + 0.30 ms vs 2.29 ms is about 1.5x) and turns into a loss at larger ones.
- Pack digests (sha256[:16]): seed 11 `df9777aad31f3945`, seed 23 `9968c750c484146b`, seed 37 `1f81f4a1d3371573`.

## Verdict
**Viable.** Querying the stored I2_S exit-trit pack directly is 5.4x to 9.0x faster for the backfill regression scan and 9.2x to 15.8x faster for per-run summary counts than int8 trits, at 4x less memory and zero result error. Caveats: absolute times are small at today's 8-step Live beta run (2 bytes per run), so the gain matters for long histories and per-test rows such as test_linux_security cases. The uint64 trick needs each run padded to a multiple of 32 trits (pad code 0 = "not exercised", which leaves pass and fail counts unchanged and is fixed up in the skip count). And it only pays if the pack is the stored form, not rebuilt for each read.

## This wave's pair files
- Pair A (JuniorOS overlay os-release.junior gaps): Trit Pack BitNet-mlx c34ba3a `scripts/os_release_junior_spec_i2s_key.py`; Winsor JuniorStock ff588ba `quant/calibration/os_release_winsor_digits_cert.py`.
- Pair B (Live beta end-to-end exit-code trits + test_linux_security): Side-Commit web3node c4dd23c `live_beta_e2e_exit_trit_ledger.py`; Handshake JuniorLLM fbfe7a1 `ports/live_beta_security_exit_trit_handshake.py`.

## Next question
For the test_linux_security summary-consistency rule (exit 0 iff failures + errors == 0 and tests_run > 0; exit 5 iff tests_run == 0), can a whole history of unittest summaries be checked from the packed per-test trits plus one exit trit per run, word-parallel, at 1.5x or better over parsing the summary text? Or, for pair A, does an os-release.junior field digest stored as trits verify faster than string compares across Debian/Alpine overlays?

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-18_1455_append-only-t2-delta-vs-fp16-int8-8gb.md.

close: 78a3cf5c8a14321a
