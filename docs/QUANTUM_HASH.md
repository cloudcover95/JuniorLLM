# Quantum hashes vs Home

There is no practical "quantum SHA" running in JuniorHome.

- Grover: preimage cost ~2^(n/2). SHA-256 ≈ 128-bit quantum security, not 0.
- Shor: breaks ECDSA/ECDH, not SHA.
- Bitcoin PoW is not "killed by Grover" in current resource estimates.
- PQ signatures (SPHINCS+, ML-DSA) *use* classical hashes; they are not trit packs.
- Quantum fingerprinting / swap-test are lab protocols, not a cache key.

Junior I2_S is lossy. It is weaker than SHA under both classical and quantum search.
Keep sha256 for integrity. Keep trit for semantic cache only.
