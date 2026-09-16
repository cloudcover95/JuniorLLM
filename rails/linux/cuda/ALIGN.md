# CUDA alignment vs Home

Home does not `nvcc` or `cudaMalloc`.
If an operator box has a GPU:
- activations: 16-byte (float4) aligned
- trit out: 16-byte aligned int8
- grid: (n+127)/128 blocks, 128 threads
- no PCIe roundtrip for T0 notes (do it on CPU)

T0/T1 stay host Winsor / junior_absmean.
