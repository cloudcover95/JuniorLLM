# Junior lattice stack (honest)

Jolt proves **RISC-V execution** under Module-SIS (~&lt;100 KB, 2M+ RV cycles/s CPU, 10M+ Apple). We do **not** beat that at RISC-V.

We prove a **different statement**: a 32-dim BitNet/MemSys vector `z ∈ {-1,0,1}^m` satisfies `A z = c (mod q)`. That is the palace/weight binding we need. Proof is tens of bytes because the witness is tens of trits, not millions of CPU cycles.

| Artifact | Status |
|----------|--------|
| Module-SIS commit | real toy (same as MemSys) |
| Polynomial / Merkle commit | real toy |
| Ternary relation argument | real toy |
| Tiny ISA + trace root | real toy, **not** RV64 |
| ML-DSA / Dilithium | **shape only**, `secure=False` |
| 128-bit PQ SNARK | **not claimed** |

`python -m lattice_zk.bench`
