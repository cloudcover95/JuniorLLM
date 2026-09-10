# Junior lattice stack

`lattice_zk/zkvm` is the BitNet-quant zkVM: 6 ops on 32 trits, AIR per step, Merkle trace, SIS on endpoints.

```
PYTHONPATH=. python -m lattice_zk.zkvm.bench
PYTHONPATH=. python tests/test_zkvm.py
```

Not RV64. Not Jolt. Not 128-bit PQ. Production-shaped for *our* statement (quant state transitions).
