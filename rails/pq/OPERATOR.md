# Operator box (not Home T0)

When you have two hosts:
1. liboqs ML-KEM-768 encapsulate for the *file* you would copy (jsonl export), not I2_S hex.
2. SPIFFE IDs for those hosts (`spiffe://junior.local/home` style). Home loopback does not issue them.
3. mTLS only on that overlay. 127.0.0.1 stays clear.

Home goldens: ml_kem false, spiffe false, mtls false.
