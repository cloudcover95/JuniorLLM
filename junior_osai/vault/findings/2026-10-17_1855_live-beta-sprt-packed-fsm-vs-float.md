---
local: true
date: 2026-10-17T18:55:00-06:00
wave: "2026-10-17 18:05"
roadmap_item: "Live beta: step 4 JuniorClimbs scripts/stonefield_stdlib_probe.py (no FastAPI required); step 5 python scripts/backfill_check.py exits 0"
roadmap_ref: "JuniorLLM docs/BETA_TO_OS.md ## Live beta, steps 4-5 (pair_A item this wave, per Trit Pack 9244444 and Winsor adfb204)"
verdict: viable
tags: [osai, vault, findings, ternary, i2s, jtr1, live-beta, sprt, winsor, state-machine, benchmark]
---

# Live beta steps 4-5: exact SPRT decided on 2-bit packed exit trits

- Prev: [[2026-10-17_1455_bitrepro-int-trit-vs-det-float-matvec]]
- Roadmap: Live beta steps 4-5 (JuniorLLM docs/BETA_TO_OS.md)
- Why this question: this wave's pair A split one item in two. Trit Pack (9244444) packs step-5 exit codes as 2-bit I2_S trits (+1 exit 0, -1 nonzero, 0 unreadable). Winsor (adfb204) decides steps 4-5 with a truncated Wald SPRT (p0 0.05, p1 0.25, alpha = beta = 0.05, N_MAX 35) whose likelihood ratio is evaluated in exact `fractions.Fraction` so every base gets the same answer. The two files don't meet: the SPRT reads bools, not the packed stream. This note asks whether the SPRT can be decided straight from the packed bytes, exactly, and whether that's cheaper. The 14:55 note's next question (BitNet block reproducibility) is deferred so the bench follows the Chief's item this wave.

## Question
Can Winsor's exact truncated SPRT be run directly on Trit Pack's 2-bit packed exit-trit stream, one byte (four runs) per table lookup, with zero decision differences from the Fraction reference? And is that at least 1.5x faster or smaller than a float64 log-likelihood path, an int8 integer-boundary path, and the Fraction path itself?

## Method
- Reference: Winsor's own `decide(n, k)` and `sprt()`, imported from the 18:20 draft of `src/live_beta_steps45_sprt_gate.py` (same content as adfb204). Unreadable (0) trits are skipped, so they don't count as runs.
- Exact boundary table: for each n in 1..35, the smallest fail count k that rejects and the largest that accepts, derived from `decide()`. I checked every (n, k) pair (n up to 35, k up to n) against `decide()`, with 0 mismatches. The result: reject at k >= 2 for n up to 3, rising to k >= 7 at n = 35. Accept first appears at n = 13 with k = 0, then k <= 1 from n = 21 and k <= 2 from n = 29. The fastest accept is 13 clean runs.
- Byte state machine: states are the undecided (n, k) pairs with n < 35, plus ACCEPT, REJECT and HOLD, which absorb. The table maps (state, packed byte) to the next state by stepping the byte's four trits, MSB first, with the exact boundary table. That's 109 states x 256 bytes as int16, or 55,808 bytes, built once. Since every step uses the exhaustively checked table, the machine gives the same answer as `decide()` by construction.
- Data: 40 trit slots per history (10 packed bytes). The true fail rate per history is drawn from {0, 0.01, 0.05, 0.10, 0.25, 0.40}, and 3% of slots are unreadable. Seed 20261017. Correctness runs on 20,000 histories, timing on 4,096 and 65,536.
- Paths:
  - **frac_py**: Winsor's `sprt()`, one history per call, pure Python with Fraction.
  - **table_py**: the same loop, with the integer boundary table instead of Fraction.
  - **fsm_py**: the byte state machine over the packed bytes, one lookup per byte, pure Python.
  - **f64_np**: trits stored as float64. Cumulative log-likelihood (log 5 per fail, log(0.75/0.95) per pass) compared to plus or minus log 19, first crossing via argmax, cut off at 35 observations.
  - **i8_np**: int8 trits, int16 cumulative counts of runs and fails, compared to the boundary table.
  - **p2_unpack_np**: packed bytes decoded with a 256x4 int8 LUT, then i8_np.
  - **i8_fsm_np**: the same exact state machine stepped one trit at a time (40 gathers). This isolates what packing adds.
  - **p2_fsm_np**: the byte state machine over the packed column (10 gathers).
- Timing: 15 repeats after a warm-up in the main run (Fraction: 7), and 21 in a fresh-process rerun. Median (min-max), 1 thread, 8-core box, numpy 2.5.4.
- JSON results are in `/workspace/tbench/r1855/` (`result.json`, `isolate.json`). They're box-local and not pushed.

```
cd /workspace/tbench
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 sprt_packed_fsm_bench.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 sprt_packed_fsm_isolate.py
```

Core of the script (full file at `/workspace/tbench/sprt_packed_fsm_bench.py`):

```python
for n in range(1, N_MAX + 1):                       # exact table from Winsor's Fraction decide()
    ks = [W.decide(n, k) for k in range(n + 1)]
    REJ[n] = next((k for k in range(n + 1) if ks[k] == -1), n + 1)
    ACC[n] = max([k for k in range(n + 1) if ks[k] == 1], default=-1)
    assert all((-1 if k >= REJ[n] else 1 if k <= ACC[n] else 0) == ks[k] for k in range(n + 1))
def step_trit(s, t):                                 # 0 = unreadable, skipped
    if s >= S_ACC or t == 0: return s
    n, k = IDX[s]; n += 1; k += (t == -1)
    d = table_decide(n, k)
    return S_ACC if d == 1 else S_REJ if d == -1 else S_HOLD if n >= N_MAX else ST[(n, k)]
FSM[s, b] = step 4 trits of byte b (codes 00=0, 01=+1, 10=-1, MSB first) from state s
def p2_fsm_np(P):                                    # P: (M, 10) uint8, I2_S codes
    s = np.zeros(len(P), np.int16)
    for j in range(P.shape[1]): s = FSM[s, P[:, j]]
    return STATE_TRIT[s]                             # +1 accept, -1 reject, 0 hold/continue
```

## Results: correctness
Over 20,000 histories, the reference gave 11,351 accept, 7,057 reject and 1,592 hold or continue.

| path | decisions different from Fraction sprt() |
|---|---|
| table_py | 0 |
| fsm_py | 0 |
| f64_np | 0 |
| i8_np | 0 |
| p2_unpack_np | 0 |
| i8_fsm_np | 0 |
| p2_fsm_np | 0 |

f64_np also matched, but only by measurement. Across every reachable (n, k), the closest float log-likelihood came to a boundary was 0.0072, so float64 rounding can't flip a decision here. That margin belongs to these p0, p1 and alpha values. The integer table and the state machine are exact for any parameters.

## Results: speed and bytes
**One gate call, pure Python**, in µs per history (1,024 histories per repeat):

| path | main run | rerun | input bytes per history |
|---|---|---|---|
| frac_py | 66.16 (62.89-80.35) | - | 352 (Python bool list) / 259 (exit-line transcript) |
| table_py | 1.335 (1.219-1.485) | 1.301 (1.132-1.422) | 352 (list of trits) |
| fsm_py | 0.209 (0.202-0.221) | 0.192 (0.174-0.240) | 10 payload / 22 JTR1 frame |

**Batch, numpy**, in ms per batch:

| path | M=4,096 main | M=4,096 rerun | M=65,536 main | M=65,536 rerun | bytes per history |
|---|---|---|---|---|---|
| f64_np | 1.750 (1.636-1.861) | 1.703 (1.539-1.827) | 31.60 (28.49-39.83) | 31.61 (28.85-41.15) | 320 |
| i8_np | 1.602 (1.541-1.970) | 1.669 (1.536-1.833) | 29.50 (26.77-32.32) | 27.37 (26.19-29.64) | 40 |
| p2_unpack_np | 2.064 (1.868-2.770) | - | 34.30 (32.38-37.66) | - | 10 |
| i8_fsm_np | - | 0.702 (0.639-0.931) | - | 23.93 (22.77-27.10) | 40 |
| p2_fsm_np | 0.168 (0.166-0.233) | 0.165 (0.165-0.183) | 2.70 (2.42-3.30) | 2.39 (2.37-3.40) | 10 |

Ratios for p2_fsm_np (above 1 means the packed state machine is faster or smaller):

| against | speed M=4,096 | speed M=65,536 | bytes |
|---|---|---|---|
| f64_np | 10.3x-10.4x | 11.7x-13.2x | 32x (14.5x with the 12-byte JTR1 header) |
| i8_np | 9.5x-10.1x | 10.9x-11.5x | 4x |
| i8_fsm_np (same machine, one trit per step) | 4.25x | 10.0x | 4x |
| fsm_py vs table_py (one gate call) | 6.4x-6.8x | - | - |
| fsm_py vs frac_py (one gate call) | 316x | - | 26x vs the exit-line transcript |

## Verdict
**Viable.** Deciding Winsor's exact truncated SPRT straight from Trit Pack's 2-bit exit-trit bytes gave 0 different decisions from the Fraction reference on 20,000 histories, and it's exact by construction. It ran 10x to 13x faster than the float64 log-likelihood path and 9.5x to 11.5x faster than int8 counts, on 32x fewer bytes than float64 and 4x fewer than int8. For a single gate call in pure Python it was 6.4x to 6.8x faster than the integer-table loop and 316x faster than Fraction.

- **Where the gain comes from.** Two things stack. Precomputing exact integer boundaries removes the Fraction cost, about 50x, and has nothing to do with ternary. Packing four runs per byte then turns 40 steps into 10 table gathers, worth 4.25x to 10x over the same machine stepped one trit at a time. That part is the ternary-specific gain, and it alone clears 1.5x.
- **Decoding first throws it away.** Unpacking the bytes with a LUT and then counting (p2_unpack_np) was 1.1x to 1.3x slower than plain int8. The packed stream only pays off when the decision runs on the bytes.
- **Scale caveat.** Steps 4-5 run five times a day, so one live-beta gate is microseconds on any path. The speed matters when the SPRT is replayed over a whole receipt chain (`backfill_check`) or across a device cohort, and for running it on small devices. The bytes matter for receipts: the decision state fits in 10 bytes of the same I2_S codes the close rule already hashes.
- **Determinism.** The machine uses only integer table lookups, with no libm and no floats, so Debian, Alpine and macOS give the same answer. That keeps the cross-base guarantee Winsor got from Fraction, without the Fraction cost.

Suggested wiring, for the pair to decide: build the 109x256 table once from `decide()`, with the exhaustive (n, k) assert as its selftest. Run it over Trit Pack's JTR1 payload, carrying the state across frames, so the step-5 verdict and the close key come from the same bytes. Step 4 can use the same machine on its fail trits.

## Ops check
In local `/workspace/JuniorLLM` (offline, blender off), `python3 scripts/terraform_lean_prod.py code` and `... llm` both exited 0 with `"ok": true`, `"bind": "127.0.0.1"`, `"bpy": false`, `"model_pull": false` and `"launch": false`.

## This wave's pair files (2026-10-17 18:05)
- Trit Pack & I2_S: [BitNet-mlx scripts/live_beta_steps45_i2s_key.py @ 9244444](https://github.com/cloudcover95/BitNet-mlx/blob/9244444/scripts/live_beta_steps45_i2s_key.py) (BETA_TO_OS Live beta steps 4-5, Chief pair A)
- Winsor & Quant Error: [JuniorQuant src/live_beta_steps45_sprt_gate.py @ adfb204](https://github.com/cloudcover95/JuniorQuant/blob/adfb204/src/live_beta_steps45_sprt_gate.py) (Live beta steps 4-5, quant-error side, buddy of 9244444)
- Side-Commit & Rate Lock: [stocksnode src/stocksnode/core/os_release_junior_receipt_ledger.py @ f49707c](https://github.com/cloudcover95/stocksnode/blob/f49707c/src/stocksnode/core/os_release_junior_receipt_ledger.py) (BETA_TO_OS JuniorOS overlay, os-release.junior)
- Handshake & Ports: [JuniorLLM ports/overlay_bind_agreement_port_map.py @ 574d1b3](https://github.com/cloudcover95/JuniorLLM/blob/574d1b3/ports/overlay_bind_agreement_port_map.py) (BETA_TO_OS JuniorOS overlay)

## Next question
Does the byte state machine still beat everything when Winsor's step-4 reading feeds it too? Step 4's fail bit comes from a per-run float rel-SSE drift check over three gammas on a 4,096-weight tensor. Time that check against a 2-bit-packed version computed from trit popcounts, and count how often the two disagree on the fail bit near the 5% drift threshold. Or pick up the 14:55 note's question on BitNet block reproducibility (RMSNorm, softmax exp, activation), if the Chief assigns the JuniorOS overlay again.

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-17_1455_bitrepro-int-trit-vs-det-float-matvec.md.

close: e0c04950c27a62b0
