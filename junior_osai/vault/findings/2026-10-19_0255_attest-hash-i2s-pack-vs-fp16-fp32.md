---
local: true
date: 2026-10-19T02:55:00-06:00
wave: "2026-10-19 02:05"
roadmap_item: "Live beta: juniorctl health/ask/night/security + test_junior_aie + test_linux_security (juniorctl security gaps: weight-pack attestation side)"
roadmap_ref: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2 juniorctl security (Quant Chief pair_A item this wave; Trit Pack JuniorLLM fbc1ad8, Winsor JuniorQuant 565379b)"
verdict: viable
tags: [osai, vault, findings, ternary, i2s, live-beta, juniorctl-security, attestation, sha256, blake2b, tamper, benchmark]
---

# juniorctl security weight attestation: hashing the I2_S pack + gamma vs fp16 and fp32 weights

- Prev: [[2026-10-18_1855_exit-trit-regress-scan-i2s-vs-int8-json]]
- Roadmap: Live beta in JuniorLLM docs/BETA_TO_OS.md, step 2 `juniorctl security`, the Chief's pair A item this wave ("juniorctl security gaps only"). Pair B had the JuniorOS overlay "No OSWorld score. No kernel blob." closing clause.
- Why this question: Winsor's file this wave (JuniorQuant 565379b, `src/live_beta_security_tamper_radius.py`) says the security check commits to the I2_S pack hash and certifies which L-inf weight tampers that hash can and cannot see. Trit Pack's key (JuniorLLM fbc1ad8, `scripts/juniorctl_security_gap_i2s_key.py`) covers the unit, seccomp, OCI flag and claim gaps but not the weights. The ternary question is whether attesting the deployed ternary artifact (gamma + 2-bit pack) at `juniorctl security` time is materially cheaper than attesting fp16 or fp32 weights, and whether it gives a more useful signal.

## Question
For one BitNet-sized linear layer, is a sha256 (or blake2b) attestation over the I2_S artifact (f32 gamma + 2-bit codes) at least 1.5x faster and smaller than the same digest over fp16 or fp32 weights, with no loss of tamper detection on changes that alter the ternary codes, and does it stay stable under float noise that does not change the codes?

## Method
- Weights: synthetic example data, Student-t df=5 scaled by 0.02, seed 19 (not a real model). Shapes 640 x 2560, 2560 x 2560, 2560 x 6912 (the BitNet 2B FFN shape). Box has about 2 GB free, so no whole-model run.
- Quant: absmean gamma (float64 mean, stored f32), codes = RoundClip(W / (gamma + 1e-5), -1, 1), I2_S code = trit + 1, four per byte MSB first. Artifact = 4-byte f32 gamma + packed bytes.
- Digest: `hashlib.new(algo).update(memoryview(buf))` over the stored bytes, sha256 and blake2b, 15 repeats each, median (min-max). Pack rebuild from fp32 masters timed separately (5 repeats).
- Sensitivity, 8 trials per shape on the fp32 masters, then re-quantize and re-hash: (a) 1-ulp f32 noise on every weight (random direction, cross-engine style re-export); (b) dense L-inf tamper uniform in +-0.01 gamma; (c) dense L-inf tamper uniform in +-RATE gamma, RATE 0.05 (Winsor's r_sec budget).
- Settings: `OMP_NUM_THREADS=1`, numpy 2.5.4, Python stdlib hashlib, 8-core box, nothing bound, no network, no installs.

Command (script on the box at `/workspace/tbench/r20261019_0255/attest_hash_bench.py`; core inline below):

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 attest_hash_bench.py 15
```

```python
import hashlib, numpy as np
EPS = np.float32(1e-5)
def quant(W):
    g = np.float32(np.mean(np.abs(W), dtype=np.float64))
    return g, np.clip(np.rint(W / (g + EPS)), -1, 1).astype(np.int8)
def pack_i2s(c):
    k = (c + 1).astype(np.uint8).reshape(-1, 4)
    return ((k[:, 0] << 6) | (k[:, 1] << 4) | (k[:, 2] << 2) | k[:, 3]).astype(np.uint8)
def artifact_i2s(W):
    g, c = quant(W)
    return np.float32(g).tobytes() + pack_i2s(c).tobytes()
digest = hashlib.sha256(memoryview(artifact_i2s(W))).hexdigest()   # vs W.astype(np.float16).tobytes(), W.tobytes()
```

## Results

Digest time, ms, median (min-max), 15 repeats:

| shape | bytes fp32 | bytes fp16 | bytes I2_S | sha256 fp32 | sha256 fp16 | sha256 I2_S | fp16 / I2_S | fp32 / I2_S | blake2b fp16 | blake2b I2_S | fp16 / I2_S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 640 x 2560 | 6,553,600 | 3,276,800 | 409,604 | 4.144 (3.654-4.332) | 2.110 (1.854-2.139) | 0.264 (0.263-0.277) | 8.0x | 15.7x | 4.648 (3.896-4.868) | 0.490 (0.483-0.630) | 9.5x |
| 2560 x 2560 | 26,214,400 | 13,107,200 | 1,638,404 | 15.134 (13.955-16.723) | 7.626 (7.067-8.511) | 0.877 (0.841-0.996) | 8.7x | 17.3x | 17.808 (15.862-19.931) | 1.948 (1.852-2.234) | 9.1x |
| 2560 x 6912 | 70,778,880 | 35,389,440 | 4,423,684 | 41.951 (39.058-45.333) | 21.165 (18.921-32.651) | 2.382 (2.297-2.598) | 8.9x | 17.6x | 48.286 (43.248-50.361) | 5.676 (5.209-6.266) | 8.5x |

Sensitivity, 8 trials per shape, re-quantized from tampered fp32 masters (counts are trials whose digest changed):

| shape | perturbation | trit flips median (max) | I2_S codes digest changed | gamma bytes changed | fp16 digest changed |
|---|---|---|---|---|---|
| 640 x 2560 | 1-ulp f32 noise | 0 (0) | 0 / 8 | 0 / 8 | 8 / 8 |
| 640 x 2560 | +-0.01 gamma | 5,136 (5,190) | 8 / 8 | 8 / 8 | 8 / 8 |
| 640 x 2560 | +-0.05 gamma | 25,764 (25,942) | 8 / 8 | 8 / 8 | 8 / 8 |
| 2560 x 2560 | 1-ulp f32 noise | 0 (0) | 0 / 8 | 0 / 8 | 8 / 8 |
| 2560 x 2560 | +-0.01 gamma | 20,410 (20,769) | 8 / 8 | 8 / 8 | 8 / 8 |
| 2560 x 2560 | +-0.05 gamma | 103,147 (103,693) | 8 / 8 | 8 / 8 | 8 / 8 |
| 2560 x 6912 | 1-ulp f32 noise | 0 (0) | 0 / 8 | 0 / 8 | 8 / 8 |
| 2560 x 6912 | +-0.01 gamma | 55,948 (56,348) | 8 / 8 | 8 / 8 | 8 / 8 |
| 2560 x 6912 | +-0.05 gamma | 279,505 (280,083) | 8 / 8 | 8 / 8 | 8 / 8 |

- Error: none. Every tamper that changed any trit or gamma changed the I2_S digest (24 of 24 per tamper size), the same as fp16. Under 1-ulp noise the I2_S digest never changed, because no trit flipped and the f32 gamma was bit-identical, while the fp16 digest changed in 8 of 8 trials. That is a false alarm on a re-export that runs identically under ternary inference.
- Memory: the attested I2_S artifact is 8.0x smaller than fp16 and 16.0x smaller than fp32.
- Artifact digests (sha256[:16], seed 19): `9bd3709c7851baa0`, `561e0c0ce5e55b9d`, `6f0d5c3ada648784`.
- Pack rebuild from fp32 masters: 3.82 (3.55-12.14), 20.26 (18.96-36.08), 97.41 (94.74-160.70) ms. That is slower than hashing the fp16 bytes, so the win only holds when the pack is the stored, served form (as it is for i2sd and bitnetd). Rebuilding the pack just to attest would be a loss.

## Verdict
**Viable.** sha256 over the stored I2_S artifact is 8.0x to 8.9x faster than over fp16 and 15.7x to 17.6x faster than over fp32 (blake2b 8.5x to 9.5x vs fp16), at 8x less data and with identical detection of every tamper that changes the ternary weights. The pack digest is also stable under 1-ulp float noise that leaves inference unchanged, where an fp16 digest raises a false alarm every time. Caveats: synthetic weights; a single layer, not a whole model (the cost scales with bytes, so the ratio should hold); Winsor's r_star shows tampers below the smallest code margin are invisible to the pack digest by design, but they also leave every trit unchanged, so ternary outputs move only through gamma, and gamma is in the digest. Absolute cost is small (2.4 ms per 2B-FFN-shaped layer), so the practical gain is that `juniorctl security` can attest every served pack on each call instead of trusting a constant.

## This wave's pair files
- Pair A (Live beta juniorctl security gaps): Trit Pack JuniorLLM fbc1ad8 `scripts/juniorctl_security_gap_i2s_key.py`; Winsor JuniorQuant 565379b `src/live_beta_security_tamper_radius.py`.
- Pair B (JuniorOS overlay "No OSWorld score. No kernel blob."): Side-Commit stocksnode f557660 `src/stocksnode/core/overlay_no_kernel_osworld_receipt_ledger.py`; Handshake JuniorLLM 8543a5e `ports/overlay_no_kernel_blob_osworld_handshake.py`.

## Next question
Can the attestation be made incremental: a per-128-trit-block Merkle tree over the I2_S pack, so a single flipped block is located and re-verified at 1.5x or better over a flat re-hash? Or, for pair B, can the staged overlay tree's "no kernel blob" magic scan run on a 2-bit pack of file-type trits faster than re-reading file headers?

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-18_1855_exit-trit-regress-scan-i2s-vs-int8-json.md.

close: f1bf4ed3e7bdf517
