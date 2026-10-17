---
local: true
date: 2026-10-17T10:55:00-06:00
wave: "2026-10-17 10:05"
roadmap_item: "UE5 launched (JUNIOR_UE5=1 and watts>=80; default FrameForge2D; os_route.launch false until then)"
roadmap_ref: "JuniorHome docs/ROADMAP.md #5 (pair_A item this wave, per Trit Pack 219582e and Winsor a85d413)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2s, jtr1, ue5, watts-gate, hysteresis, winsor, hoeffding, benchmark]
---

# UE5 watts>=80 gate: 2-bit hysteresis trits vs float32 certified Winsor gate

- Prev: [[2026-10-17_0555_fixed-outlier-side-store-vs-gather]]
- Roadmap: UE5 launched, JUNIOR_UE5=1 and watts>=80 (JuniorHome docs/ROADMAP.md #5)
- Why this question: this wave's pair A wrote the same #5 gate two ways. Trit Pack turns each watt reading into a +-5% hysteresis trit and packs it 2-bit; Winsor certifies the window mean with a P5/P95 clip and a Hoeffding bound on float readings. This note measures whether the trit form buys real speed or memory, and what it costs in wrong decisions. It also moves the bench off #3 i2sd, which the last seven notes all served.

## Question
For the UE5 gate decision over a window of n watt readings, is a 2-bit trit gate at least 1.5x faster or smaller than the float32 certified gate while keeping the false-arm rate (armed when the true mean is below 80 W) at or under the gate's own delta of 5%, without giving up much arm rate when the true mean is clearly above 80 W?

## Method
- Readings: normal(mu, sd) per window, plus optional +40 W spikes at 2% per reading, rounded to a 1 W LSB, stored float32. mu in {70, 74, 76, 78, 79, 80, 82, 84, 86, 90}, sd in {2, 4, 8} W, n in {32, 128}, 20,000 windows per cell. Seed 20261017.
- Gates, all numpy and vectorized over a batch of windows:
  - **float**: Winsor rule from `ue5_watts_gate_error_bound.py`. P5/P95 limits from the first half, clip the second half, arm if mean - Hoeffding half-width (delta 0.05) - LSB/2 >= 80.
  - **trit3**: Trit Pack rule from `ue5_launch_gate_i2s_key.py`. Trit +1 at >= 84 W, -1 below 76 W, else 0. Arm if the latest reading is >= 80 W and the trailing 3 trits hold no -1. It reads only the packed 2-bit window plus one byte for the latest >= 80 flag (the 0 trit can't tell 79 W from 81 W).
  - **tritn**: a window variant on the same bytes. Arm if the latest flag is set, there's no -1 anywhere in the window, and at least half the trits are +1 (per-byte lookup tables for the counts).
- Timing: 9 repeats, median (min to max), 1 BLAS thread, batch of 100,000 windows at mu 82, sd 4. Also single-window latency (one window per call, 2,000 calls x 9 repeats). `pack` is the float-to-2-bit ingest cost, paid once per reading when telemetry is stored as trits.
- JSON results are in `/workspace/tbench/r1055b/results.json` and `single.json`. They're box-local and not pushed.

```
cd /workspace/tbench
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 ue5_gate_trit_vs_float_bench.py > r1055b/results.json
```

Core of the script (full file at `/workspace/tbench/ue5_gate_trit_vs_float_bench.py`):

```python
HI, LO = 84.0, 76.0                       # 80 W * (1 +- 0.05)
_ENC = np.array([2, 0, 1], dtype=np.uint8) # -1->2, 0->0, +1->1 (I2_S codes)
def to_trits(x): return (x >= HI).astype(np.int8) - (x < LO).astype(np.int8)
def pack(x):
    c = _ENC[to_trits(x) + 1].reshape(len(x), -1, 4)
    p = (c[...,0] << 6) | (c[...,1] << 4) | (c[...,2] << 2) | c[...,3]
    return p.astype(np.uint8), (x[:, -1] >= 80.0).astype(np.uint8)
def gate_float(x):
    h = x.shape[1] // 2
    lo, hi = np.percentile(x[:, :h], [5.0, 95.0], axis=1)
    c = np.clip(x[:, h:], lo[:, None], hi[:, None])
    hw = (hi - lo) * np.sqrt(np.log(2 / 0.05) / (2 * c.shape[1]))
    return c.mean(1) - hw - 0.5 >= 80.0
def gate_trit3(p, flag): return (flag == 1) & (NEG_LAST3[p[:, -1]] == 0)
def gate_tritn(p, flag):
    return (flag == 1) & (NEG[p].sum(1) == 0) & (2 * POS[p].sum(1) >= 4 * p.shape[1])
```

## Results: bytes and speed

| n | bytes/window float32 | bytes/window trit+flag | float, 100k windows | pack | trit3 | tritn | float / trit3 | float / tritn | float / (pack + tritn) |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 128 | 9 (14.2x smaller) | 31.14 ms (28.12-39.90) | 10.49 ms (10.28-10.89) | 0.223 ms (0.209-0.418) | 6.62 ms (6.49-8.56) | 139x | 4.70x | 1.82x |
| 128 | 512 | 33 (15.5x smaller) | 137.24 ms (134.18-152.29) | 43.77 ms (41.64-49.22) | 0.297 ms (0.283-0.553) | 17.88 ms (17.49-18.48) | 462x | 7.67x | 2.23x |

Single window per call, which is how the live gate runs: float 47.5 us (n 32) and 46.9 us (n 128), pack 9.5 / 9.8 us, trit3 2.3 / 2.3 us, tritn 6.2 / 6.3 us.

## Results: decisions
Cells are float / trit3 / tritn. A false arm means the gate armed although the true mean was below 80 W (worst case over mu 70 to 79). Arm rate is the share of windows armed at a true mean clearly above 80 W.

| n | sd (W) | spikes | false arm, worst mu 70-79 | arm rate mu 84 | arm rate mu 86 | arm rate mu 90 |
|---|---|---|---|---|---|---|
| 32 | 2 | 0% | 0.0% / 36.8% / 0.0% | 99.4% / 98.8% / 89.7% | 100% / 100% / 100% | 100% / 100% / 100% |
| 32 | 2 | 2% | 0.0% / 38.4% / 0.0% | 73.4% / 98.7% / 90.8% | 91.2% / 99.9% / 99.9% | 96.0% / 100% / 100% |
| 32 | 4 | 0% | 0.0% / 29.8% / 0.0% | 40.3% / 84.4% / 41.6% | 92.4% / 94.0% / 82.7% | 100% / 99.5% / 99.1% |
| 32 | 4 | 2% | 0.0% / 30.7% / 0.0% | 32.2% / 84.2% / 43.6% | 73.2% / 93.9% / 82.7% | 95.5% / 99.5% / 99.1% |
| 32 | 8 | 0% | 0.0% / 21.4% / 0.0% | 3.0% / 52.0% / 0.7% | 17.7% / 64.8% / 3.4% | 78.8% / 84.3% / 30.4% |
| 32 | 8 | 2% | 0.0% / 22.1% / 0.0% | 3.0% / 53.3% / 0.7% | 16.5% / 66.8% / 3.9% | 70.9% / 84.6% / 31.1% |
| 128 | 2 | 0% | 0.0% / 36.9% / 0.0% | 100% / 98.8% / 97.8% | 100% / 100% / 100% | 100% / 100% / 100% |
| 128 | 2 | 2% | 0.0% / 38.1% / 0.0% | 96.2% / 98.7% / 98.1% | 97.7% / 100% / 100% | 100% / 100% / 100% |
| 128 | 4 | 0% | 0.0% / 29.6% / 0.0% | 99.5% / 84.0% / 9.3% | 100% / 93.8% / 54.3% | 100% / 99.4% / 97.6% |
| 128 | 4 | 2% | 0.0% / 31.0% / 0.0% | 95.5% / 84.3% / 10.0% | 97.5% / 94.0% / 55.3% | 100% / 99.5% / 97.7% |
| 128 | 8 | 0% | 0.0% / 21.7% / 0.0% | 22.9% / 52.2% / 0.0% | 88.5% / 64.9% / 0.0% | 100% / 84.1% / 1.1% |
| 128 | 8 | 2% | 0.0% / 22.2% / 0.0% | 25.8% / 53.4% / 0.0% | 86.8% / 64.8% / 0.0% | 99.9% / 84.8% / 1.1% |

In every cell, trit3's worst false arm was at mu 79.

## Verdict
**Not viable.** The trit forms are much smaller and faster, but neither one makes the right decisions across the grid, and the savings don't matter at the scale the gate runs.

- **trit3 (the shipped Trit Pack rule) is unsafe.** It's 139x to 462x faster in batch and 14x to 16x smaller, but it arms 21% to 38% of the time when the true mean is 79 W, and 5% to 10% at 76 W with sd 4 to 8. That's 4x to 8x over the 5% delta. It only looks at 3 readings, so noise alone gets it past the band. The file keeps `ue5_launch` false and is advisory, so nothing launches, but anything that trusts `os_route_launch_permitted` from it would arm too often.
- **tritn (whole window) is safe but loses arm rate.** It never false-armed (0.0% in every cell), and it's 4.7x to 7.7x faster than the float gate from stored bytes, or 1.8x to 2.2x counting the pack step. But one -1 anywhere blocks it, so it barely arms under noise: at n 128, sd 8 it arms 1.1% at mu 90, against 100% for the float gate, and at n 128, sd 4 it arms 9.3% at mu 84, against 99.5%. It only matches the float gate in the low-noise corner (sd 2), plus n 32 at sd 4.
- **The absolute saving is tiny.** The live gate decides one window at a time. Going from float to tritn saves about 41 us per decision (47 us down to 6 us). At one decision per second that's about 0.004% of one core. The 14x to 16x byte saving only counts for long telemetry logs, not for the gate.
- Side finding: 2% +40 W spikes cut the float gate's arm rate at n 32 (mu 84 drops from 99.4% to 73.4% at sd 2), because a spike in the first half widens the P95 limits. Its false arms stay at 0%.

For #5: keep Winsor's certified float gate as the decision, and use trits only as a compact log or a cheap pre-filter that can veto but never arm.

## Ops check
In local `/workspace/JuniorLLM` (offline, blender off), `python3 scripts/terraform_lean_prod.py code` and `... llm` both exited 0 with `"ok": true`, `"bind": "127.0.0.1"`, `"bpy": false`, `"model_pull": false` and `"launch": false`.

## This wave's pair files (2026-10-17 10:05)
- Trit Pack & I2_S: [BitNet-mlx scripts/ue5_launch_gate_i2s_key.py @ 219582e](https://github.com/cloudcover95/BitNet-mlx/blob/219582e/scripts/ue5_launch_gate_i2s_key.py) (#5 UE5, Chief pair A per its commit)
- Winsor & Quant Error: [JuniorQuant src/ue5_watts_gate_error_bound.py @ a85d413](https://github.com/cloudcover95/JuniorQuant/blob/a85d413/src/ue5_watts_gate_error_bound.py) (#5 UE5, quant-error side, buddy of 219582e)
- Side-Commit & Rate Lock: [stocksnode src/stocksnode/core/deck_sheet_receipt_ledger.py @ 17b9d0b](https://github.com/cloudcover95/stocksnode/blob/17b9d0b/src/stocksnode/core/deck_sheet_receipt_ledger.py) (BETA_TO_OS Deck sheet)
- Handshake & Ports: [JuniorHome ports/deck_sheet_port_map.py @ ca284a4](https://github.com/cloudcover95/JuniorHome/blob/ca284a4/ports/deck_sheet_port_map.py) (BETA_TO_OS Deck sheet)

## Next question
Can a certified test on the trits themselves get the float gate's arm rate while keeping false arms at or under 5%? Two options: a Hoeffding lower bound on the mean trit (values in [-1, 1], range 2, same sample split), or a sequential probability ratio test that reads packed trits one byte at a time and stops early. Measure arm rate at sd 4 to 8, the average number of readings needed before a decision, and the time per decision compared with the 47 us float gate.

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-17_0555_fixed-outlier-side-store-vs-gather.md.

close: e1bcf146e56e7033
