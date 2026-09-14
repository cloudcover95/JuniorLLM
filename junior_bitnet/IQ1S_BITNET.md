# IQ1_S vs BitNet I2_S vs Junior wire

## IQ1_S (ggml)
~1.56 bpw. Superblock 8×32. 11-bit index into a 2048-entry **ternary grid**, plus 3-bit scale and a delta. Needs an importance matrix at quantize time. Dequant is `d*(2*ls+1)*(grid[j]+delta)`, not clip(round(W/γ)).
Not our handshake pack.

## BitNet b1.58 / I2_S
Weights trained into {-1,0,1}. GGUF dtype often 36. Block 128 weights / 32 bytes (2 bits each). bitnet.cpp codes: -1→00, 0→01, 1→10. Matmul is add/sub/skip. Official 2B is a **file** (`bitnet.cpp` / I2_S GGUF). T3 reads it if present. Home does not download it.

## Junior T0 wire
Winsor-p95 on a **note**. I2_S pack here: 0→0, 1→1, -1→2 (two bits). Same alphabet, different codebook than bitnet.cpp. Do not mix bytes.

TQ1_0/TQ2_0 in llama.cpp are the in-tree ternary GGUF types (~1.69 / 2.06 bpw).
