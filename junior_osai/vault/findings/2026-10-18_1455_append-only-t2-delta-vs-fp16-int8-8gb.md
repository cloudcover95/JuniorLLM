---
local: true
date: 2026-10-18T14:55:00-06:00
wave: "2026-10-18 14:05"
roadmap_item: "Live beta step 6: No deletes. No >8GB pull. bitnetd on 127.0.0.1:8765 only (no-deletes and 8GB-pull side)"
roadmap_ref: "JuniorLLM docs/BETA_TO_OS.md Live beta step 6 (Quant Chief pair_A item this wave; Trit Pack JuniorLLM c65b936, Winsor JuniorQuant eaccb94)"
verdict: viable
tags: [osai, vault, findings, ternary, i2s, jtr1, winsor, live-beta, step6, no-deletes, pull-cap, delta, benchmark]
---

# Append-only version store: 2-bit ternary delta vs full repack, fp16 and int8 deltas under the 8 GB pull cap

- Prev: [[2026-10-18_1055_kv-cache-t2-vs-fp16-int8-memmax]]
- Roadmap: Live beta step 6, "No deletes. No >8GB pull. bitnetd on `127.0.0.1:8765` only" (JuniorLLM docs/BETA_TO_OS.md), the Chief's pair A item this wave.
- Why this question: Trit Pack's step 6 key (c65b936) turns "no deletes" and "no >8GB pull" into guard trits, and Winsor's bound (eaccb94) argues a pack is only append-only safe if every engine re-derives the same gamma. Neither says what append-only costs. If nothing is ever deleted, every new weight version is appended, and a client that already holds the last version should pull only the change. The ternary question is whether a 2-bit trit delta makes that history cheap enough to matter, compared with a full repack and with fp16 or int8 deltas.

## Question
For a weight update appended under "no deletes", is an I2_S trit delta (XOR of the 2-bit packs, zlib, or a sparse flip list) at least 1.5x smaller or faster than a full 2-bit repack and than fp16 or int8 deltas, while reconstructing the new pack bit for bit? How many versions fit under the 8 GB pull cap for the 2.4B-param pack (600,000,004 bytes, from Trit Pack's 10:20 key)?

## Method
- Weights: one 2048 x 4096 matrix (8,388,608 weights), Student-t (df 5) x 0.02 with lognormal per-row scale, seeds 11, 23 and 37. Version 2 is version 1 plus an update: dense Gaussian noise at 1e-3, 1e-2 and 5e-2 of the weight std, or 5e-2 on a random 1% of rows (a sparse fine-tune).
- Ternary: absmean gamma, trit = RoundClip(W / (gamma + 1e-6), -1, 1), I2_S code trit + 1, four per byte MSB first, plus a 4-byte f32 gamma. Gamma either re-derived on version 2 or pinned to version 1 (Winsor's append-only condition).
- Deltas: fp16 XOR delta + zlib 6; int8 (per-row absmax, fp16 scales) XOR delta + zlib 6 for codes and scales; trit XOR delta of the packed bytes + zlib 6; trit sparse delta (zlib of uint32 index gaps + new 2-bit codes). Full-version baselines: fp16 16 bits/weight, int8 8.0039, I2_S 2.0, I2_S + zlib.
- Checks: every delta is applied back and compared byte for byte (fp16 and trit XOR in the timing run, sparse in `sparse_verify.txt`, pack round trip asserted on every seed).
- Timing on the 1e-2 dense update with pinned gamma: encode (XOR + zlib), apply (unzip + XOR), sha256 of the full version, and a full re-quantize + repack. 7 repeats, median (min-max), one thread, numpy 2.5.4, 8-core box, nothing bound, no network.
- Box-local results: `/workspace/tbench/r20261018_1455/results.json`, `versions.txt`, `sparse_verify.txt` (not pushed).

```
cd /workspace/tbench/r20261018_1455
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 append_delta_bench.py 7 > results.json
```

Core of the script (full file at `/workspace/tbench/r20261018_1455/append_delta_bench.py`):

```python
def tern(W, gamma=None):
    g = np.abs(W).mean() if gamma is None else gamma
    return np.clip(np.rint(W / (g + EPS)), -1, 1).astype(np.int8), float(g)

def pack_i2s(t):                  # code = trit + 1, four per byte, MSB first
    c = (t.reshape(-1, 4) + 1).astype(np.uint8)
    return (c[:, 0] << 6 | c[:, 1] << 4 | c[:, 2] << 2 | c[:, 3]).astype(np.uint8)

t1, g1 = tern(W1); t2, _ = tern(W2, g1)              # gamma pinned to v1
p1, p2 = pack_i2s(t1.reshape(-1)), pack_i2s(t2.reshape(-1))
delta = zlib.compress(np.bitwise_xor(p1, p2).tobytes(), 6)
assert np.bitwise_xor(p1, np.frombuffer(zlib.decompress(delta), np.uint8)).tobytes() == p2.tobytes()
```

## Results

Bytes per appended version, bits per weight, median of 3 seeds (seed spread under 1% on every dense row, under 13% on the sparse row):

| update | trits flipped | fp16 XOR+z | int8 XOR+z | I2_S full | I2_S full+z | trit XOR+z | trit sparse | trit delta vs I2_S full | vs int8 delta | vs fp16 delta |
|---|---|---|---|---|---|---|---|---|---|---|
| dense 1e-3 | 0.073% | 7.04 | 0.265 | 2.00 | 1.60 | 0.0205 | 0.0143 | 140x | 18.5x | 492x |
| dense 1e-2 | 0.73% | 10.79 | 1.335 | 2.00 | 1.60 | 0.1237 | 0.1106 | 18.1x | 12.1x | 97.6x |
| dense 5e-2 | 3.67% | 12.83 | 3.224 | 2.00 | 1.60 | 0.3869 | 0.4247 | 5.2x | 8.3x | 33.2x |
| 1% rows 5e-2 | 0.035% | 0.154 | 0.043 | 2.00 | 1.60 | 0.0063 | 0.0043 | 465x | 10.0x | 35.8x |

(The ratio columns use the smaller of the two trit deltas.) Pinned and re-derived gamma gave the same flip share on every row: gamma moved from 0.0198329 to 0.0198344 (relative 7.6e-5) under these symmetric updates.

Versions that fit in one 8 GB pull for the 2.4B-param model (base plus appended deltas, extrapolated linearly from bits per weight):

| update | fp16 (4.8 GB base) | int8 (2.40 GB base) | I2_S full repacks | I2_S + trit delta |
|---|---|---|---|---|
| dense 1e-3 | 2 | 71 | 13 | 1,721 (4.3 MB per delta) |
| dense 1e-2 | 1 | 14 | 13 | 224 (33.2 MB per delta) |
| dense 5e-2 | 1 | 6 | 13 | 64 (116.1 MB per delta) |
| 1% rows 5e-2 | 70 | 432 | 13 | 5,700 (1.3 MB per delta) |

Speed on 8,388,608 weights, dense 1e-2, pinned gamma, ms, median (min-max of 7):

| step | fp16 | ternary I2_S | ternary gain |
|---|---|---|---|
| encode delta (XOR + zlib) | 1430.07 (1421.31-1473.88) | 32.20 (29.69-40.75) | 44x |
| encode sparse delta | | 30.70 (29.22-41.55) | |
| apply delta (unzip + XOR) | 108.41 (99.72-123.52) | 2.87 (2.74-3.30) | 37.7x |
| sha256 of full version | 10.68 (9.79-11.20) | 1.41 (1.29-1.48) | 7.6x |
| full re-quantize + repack | | 19.10 (18.83-46.70) | |

Error: every delta reconstructs its version bit-identically (0 differing bytes; sparse check flipped 61,635 trits and matched). The schemes' own quantization error on these synthetic float weights is fp16 0.0002, int8 0.020 and ternary 0.59 relative; that is the cost of ternarizing weights that were not trained ternary and is the same whether a version is stored full or as a delta, so it is not part of this comparison.

## Verdict
**Viable, for storage and pull size under "no deletes".** Appending each new version as a 2-bit trit delta instead of a full I2_S repack is 18x smaller at a 1e-2 update (33.2 MB vs 600 MB per version for the 2.4B pack), still 5.2x at 5e-2, and 12x smaller than the int8 delta and 98x smaller than the fp16 delta at the same update, with bit-identical reconstruction. Under the 8 GB pull cap that is 224 versions of history against 13 full repacks, 14 int8 versions and a single fp16 version. Applying a delta is 2.9 ms per 8.4M weights against 108 ms for fp16 (37.7x); extrapolated to 2.4B weights that is about 0.8 s against 31 s.

- **The gain shrinks with update size.** Flip share grows about linearly with the update (0.73% of trits per 1e-2 std), and by linear extrapolation (about 10.5 delta bits per flipped trit at 3.7% flips) the delta stays 1.5x under a full 2-bit repack only until about 13% of trits flip; a larger retrain should be appended as a full repack.
- **Sparse beats XOR+zlib only for small updates.** The index-gap list wins below about 1% flips and loses at 3.7%, so a writer can pick whichever is smaller per version.
- **For Winsor's bound.** Gamma drift did not cause flips here; flips track the update itself. Pinning gamma per pack, as Winsor's f64-fsum row recommends, costs nothing in delta size and keeps reconstruction engine-independent.
- **Caveat.** Synthetic weights and Gaussian updates; a real ternary fine-tune may flip trits in more clustered places, which would help zlib, or flip more of them per step, which would not. The per-version numbers above should be re-measured on the first real pair of packs.

Suggested wiring, for the pair to decide: Trit Pack's step 6 key could accept a delta manifest line ("<bytes> <name> delta-of <sha>") so a delta counts against the 8 GB total at its own size, and Winsor's file could add a row that checks a delta's flip share against the full-repack crossover.

## Ops check
In local `/workspace/JuniorLLM` (offline, blender off), `python3 scripts/terraform_lean_prod.py code` and `... llm` both exited 0 with `"ok": true`, `"bind": "127.0.0.1"`, `"bpy": false`, `"model_pull": false` and `"launch": false`.

## This wave's pair files (2026-10-18 14:05)
- Chief assignment: pair_A Live beta step 6, no deletes, no >8GB pull, bitnetd 127.0.0.1:8765 only; pair_B JuniorOS overlay confirm receipts (no Chief file for this wave on the box; taken from the pairs' headers and commits).
- Trit Pack & I2_S: [JuniorLLM scripts/live_beta_step6_guard_i2s_key.py @ c65b936](https://github.com/cloudcover95/JuniorLLM/blob/c65b936/scripts/live_beta_step6_guard_i2s_key.py) (step 6 guard trits over diff, size manifest and ss captures, I2_S key)
- Winsor & Quant Error: [JuniorQuant src/live_beta_step6_gamma_repro_bound.py @ eaccb94](https://github.com/cloudcover95/JuniorQuant/blob/eaccb94/src/live_beta_step6_gamma_repro_bound.py) (cross-engine Winsor gamma reproducibility bound for no-delete packs)
- Side-Commit & Rate Lock: [stocksnode src/stocksnode/core/bitnetd_user_unit_parity_receipt_ledger.py @ 43ff141](https://github.com/cloudcover95/stocksnode/blob/43ff141/src/stocksnode/core/bitnetd_user_unit_parity_receipt_ledger.py) (bitnetd user-unit hardening parity receipts)
- Handshake & Ports: [JuniorLLM ports/os_release_junior_reader_handshake.py @ 973ea64](https://github.com/cloudcover95/JuniorLLM/blob/973ea64/ports/os_release_junior_reader_handshake.py) (os-release.junior reader/writer handshake and overlay confirm receipt)

## Next question
Does a chain of appended trit deltas stay cheap to replay? Measure the time and peak memory to rebuild version k from the base plus k deltas against keeping a full repack every m versions (a checkpoint), and find the m that keeps a cold start under one second for the 2.4B pack. Or take the 10:55 note's mixed int8-K / int4-V cache question if the Chief moves on.

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-18_1055_kv-cache-t2-vs-fp16-int8-memmax.md.

close: f0ae8137a616b8d0
