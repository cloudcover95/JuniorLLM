# Trit neighbor hashing

Each trit is stored as two bits: \(t+1\) so \(-1\mapsto 00,\ 0\mapsto 01,\ 1\mapsto 10\). Pattern `11` is unused.
Hex of that bitstream is `i2s_hex`.

Hamming on **trits** is the real neighborhood: one sign flip → distance 1.
Hamming on **packed bits** over-weights \(+1\leftrightarrow -1\) (two bits flip: `10` vs `00`).
`ports.ham_pq.ham` is bit Hamming. Prefer trit Hamming when comparing shoes/notes.

Not a hash: many notes map to the same pack (Winsor collapse, all-positive ids).
SHA3-256 sits beside it for exact identity (`ports.sha3_note`, `ham_pq` already tags SHA3).
