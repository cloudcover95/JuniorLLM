# Lattice KEM vs Junior wire

NIST ML-KEM (FIPS 203, Kyber family): ML-KEM-512/768/1024. Module-LWE. Encapsulate → ciphertext + shared secret.
Home does not implement it. Do not roll your own NTT.

Operator onboard: liboqs / boringssl ML-KEM when you need TLS 1.3 PQ.
Trit I2_S is not a KEM. Hamming cache is not encapsulate.

Related: ML-DSA (FIPS 204) signatures, SLH-DSA (FIPS 205) hash signatures.
