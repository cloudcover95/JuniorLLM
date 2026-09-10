# bitnet_pq — BitNet × quant chain group

**Target:** 128-bit post-quantum argument + a chain VM that only accepts BitNet-quant state transitions.
**Reality:** `Params.secure` is `False`. Challenges are 128-bit wide. Soundness is not a published reduction.

Do not take deposits. Do not replace Dilithium/Jolt.

```
PYTHONPATH=. python -m bitnet_pq.scripts.cli genesis
PYTHONPATH=. python -m bitnet_pq.scripts.cli tick
PYTHONPATH=. python -m bitnet_pq.scripts.cli bench
PYTHONPATH=. python tests/test_bitnet_pq.py
```
