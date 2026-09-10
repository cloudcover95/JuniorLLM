# What we have vs a16z Jolt Module-SIS

Structure already in JuniorCloud:

- BitNet ternary `{-1,0,1}` (FieldCore, NightTernary, rigid IQ)
- MemSys palace + bit-drift (JuniorMemSys-Suite + `adaptations/astra_reason/memsys_bridge.py`)
- Hash-pin / guardrails / STATE receipts
- No elliptic-curve proofs. No Jolt. No RISC-V zkVM.

Jolt's move is a **prover** on Module-SIS. We reuse the *short-vector* alphabet as a palace commitment in MemSys `lattice/msis_commit.py`.
Do not advertise 2e6 RISC-V cycles/s or 128-bit PQ SNARKs.
