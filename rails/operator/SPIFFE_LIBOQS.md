# Operator box (not Home T0)

When you have two hosts:
1. liboqs ML-KEM-768 encapsulate for session bytes (jsonl export), not I2_S keys.
2. SPIFFE IDs for those hosts. SVID ≠ trit hex.
3. mTLS on the wire between boxes. Home stays 127.0.0.1.

Home goldens: spiffe false, ml_kem false, mtls false.
