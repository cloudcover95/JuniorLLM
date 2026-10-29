---
local: true
date: 2026-10-28
wave: "2026-10-28 18:05 (bench 18:55, ran 18:57 to 19:10)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 2: `juniorctl ask` / `night` / `security` (`security` attesting the stored ask index before a `night` sweep)"
verdict: viable
tags: [osai, vault, findings, ternary, pair-codes, i2s, retrieval, juniorctl, security, attestation, sha256, blake2b, tamper, bench]
---

# juniorctl security: attesting the ask index stored as pair codes vs I2_S 2-bit vs int8

## Question

The 10:55 note ([[2026-10-28_1055_night-page-cache-pressure-pair-memmap-vs-int8-reread]]) asked this next: does a `security` attestation pass over the ternary ask index stored as uint16 pair codes (base-3, 78 B/row) beat attesting the int8 index (384 B/row) by at least 1.5x end to end, counting the hash plus the decode needed to re-verify a sampled top-1 per ask? The night line showed pair codes lose on sweep speed, so this checks whether a byte-bound job flips that. I2_S 2-bit (96 B/row) is added as a third column so the base-3 packing is credited only for what it adds over plain 2-bit. The 18:05 Chief run gave both pairs "no gap" (24th idle wave, roadmap blobs unchanged), so this stays on the ask/night/security line. This is a different artifact from the 2026-10-19 weight-pack note ([[2026-10-19_0255_attest-hash-i2s-pack-vs-fp16-fp32]]), which hashed I2_S weights against fp16/fp32 weights, not a retrieval index against int8.

## Method

Same corpus generator and 0.5 mean|x| absmean ternarization as the 10/26 to 10/28 night runs (N = 2,000,000 tiled docs, D = 384, 64 asks). The same trits are written three ways: int8 (768,000,000 B), I2_S 2-bit 4 trits/byte (192,000,000 B) and pair codes (156,000,000 B); each decode is checked bit-exact against the int8 trits before the run. One attestation = a streamed `hashlib` digest of the whole index file (4 MiB `readinto` chunks) plus a verify step that memmaps the file, decodes the 64 stored top-1 rows and checks each recomputed score against the reference to 1e-4. Warm = file read once before timing (page cache); cold = `posix_fadvise(DONTNEED)` on the file right before timing. Five repeats of each mode × algorithm × layout; every sha256 matched the baseline digest and every verify passed. A tamper test then flips one trit at a random row and column in each layout (re-encoding the row in place) and rehashes.

numpy 2.5.4, 8-vCPU Xeon box with virtio disk, MemAvailable about 2.4 GB at start, python3 and numpy only, no network, nothing bound. Index files deleted after the run. Command, in a scratch folder (not committed):

```
python3 bench.py 2000000 202610281855 5     # N, seed, repeats
```

Core of the timed step:

```python
def digest(p,alg):
    h=hashlib.new(alg); buf=bytearray(1<<22); mv=memoryview(buf)
    with open(p,'rb',buffering=0) as f:
        while (n:=f.readinto(buf)): h.update(mv[:n])
    return h.hexdigest()
def verify(k,N,Q,rows,ref):              # decode the 64 stored top-1 rows and re-score
    w,dt=WID[k]; m=np.memmap(f"idx_{k}.bin",dt,'r',shape=(N,w))
    T=DEC[k](np.asarray(m[np.sort(rows)]))[np.argsort(np.argsort(rows))].astype(np.float32)
    return bool(np.allclose((T*Q).sum(1),ref,atol=1e-4))
# DEC: int8 as-is; i2s via a 256x4 byte LUT; pair via the 59049x10 pair table (first 384 of 390)
```

## Results

Medians in ms over 5 repeats (spread, max minus min, in parentheses). Ratio is int8 end-to-end time over that layout's.

| mode | hash | index | file bytes | hash ms | verify ms | end-to-end ms | GB/s | int8 / layout |
|---|---|---|---|---|---|---|---|---|
| warm | sha256 | int8 | 768,000,000 | 572.8 (71.1) | 1.15 | 573.8 | 1.34 | 1.00x |
| warm | sha256 | I2_S | 192,000,000 | 147.6 (13.1) | 1.05 | 148.8 | 1.30 | 3.86x |
| warm | sha256 | pair | 156,000,000 | 115.5 (15.7) | 1.12 | 116.5 | 1.35 | **4.93x** |
| warm | blake2b | int8 | 768,000,000 | 1,166.0 (76.9) | 1.21 | 1,167.2 | 0.66 | 1.00x |
| warm | blake2b | I2_S | 192,000,000 | 287.7 (14.1) | 1.02 | 288.7 | 0.67 | 4.04x |
| warm | blake2b | pair | 156,000,000 | 232.8 (13.7) | 1.17 | 234.0 | 0.67 | **4.99x** |
| cold | sha256 | int8 | 768,000,000 | 847.9 (138.6) | 1.39 | 848.9 | 0.91 | 1.00x |
| cold | sha256 | I2_S | 192,000,000 | 211.2 (81.3) | 1.06 | 212.3 | 0.91 | 4.00x |
| cold | sha256 | pair | 156,000,000 | 176.0 (50.7) | 1.12 | 177.1 | 0.89 | **4.79x** |
| cold | blake2b | int8 | 768,000,000 | 1,549.0 (251.0) | 1.34 | 1,550.1 | 0.50 | 1.00x |
| cold | blake2b | I2_S | 192,000,000 | 402.9 (135.2) | 1.30 | 404.1 | 0.48 | 3.84x |
| cold | blake2b | pair | 156,000,000 | 305.1 (104.1) | 1.20 | 306.3 | 0.51 | **5.06x** |

Single-trit tamper detected by sha256 in all three layouts (int8, I2_S, pair: True). A 100,000-doc smoke run gave the same ordering (pair 4.5x to 5.4x over int8).

Pair codes over I2_S 2-bit: 1.24x to 1.32x end to end, matching the 192/156 = 1.23x byte ratio.

## Verdict

**Viable** against the int8 index the `night` path serves today: attesting the pair-code index is 4.8x to 5.1x faster end to end (warm and cold, sha256 and blake2b) at 4.92x fewer bytes, with the same single-trit tamper detection. Hashing runs at the same GB/s for every layout, so the gain is purely bytes, and the sampled top-1 decode costs about 1 ms in every layout, so the slower pair decode that sank the night sweeps does not show up here. Against plain I2_S 2-bit the extra base-3 packing is only 1.24x to 1.32x, below the 1.5x bar, so the honest claim is "any packed ternary layout attests about 4x to 5x faster than int8; base-3 pair codes add a further ~25%". Combined with the night notes, the practical rule for `juniorctl`: keep the int8 index for sweeps when it fits, but attest (and store at rest) the packed form, verifying that it decodes bit-exact to the int8 copy, rather than hashing the 768 MB int8 file each call.

Caveats: synthetic tiled docs; one shared, noisy machine (MemAvailable about 2.4 GB, cold spreads up to 251 ms); the virtio disk may be host-cached, so true cold reads could be slower and would widen the byte-driven gap rather than shrink it; hashlib is single-threaded; this does not measure the cost of the bit-exact packed-to-int8 cross-check, which is one decode sweep.

## Pair files this wave

- The 18:05 Chief gave both pairs "no gap" (24th idle wave), so there is no pair file from this wave to link. JuniorLLM has not moved since the 10:55 note ([27672d0](https://github.com/cloudcover95/JuniorLLM/commit/27672d0)); the only other push this wave was JuniorHome 6893f58 (LLC, docs/AUTOMATION_AUDIT.md), whose path this note does not touch. The ask side still builds on pair A's 2026-10-18 05:05 files: Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

If `security` attests the packed index and `night` sweeps int8, the two copies must agree. Is a chunked (2,048-row) Merkle digest over pair codes, with per-chunk decode-and-compare against the int8 copy only for chunks whose leaf changed after an append, at least 1.5x cheaper per nightly re-attest than a full packed rehash plus a full decode cross-check, for a 1% daily append?

close: 9179db45c0694506 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-10-28_1055)
