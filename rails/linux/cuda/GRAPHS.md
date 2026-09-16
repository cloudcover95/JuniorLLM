# Tensor cores / CUDA Graphs vs Junior wire

Tensor cores (WMMA 16×16×16 FP16/TF32/INT8) want dense tiles.
Our pack is {-1,0,1} add/sub/skip. That is not an HMMA shape.

CUDA Graphs help when the same kernel is launched thousands of times with fixed buffers.
Handshake is once per note, n ≪ 1e5. Graph capture + instantiate costs more than the pack.

Use if all of:
- n ≥ 1e5
- same shape reused ≥ 1e3 times
- bench_pipe shows host pack is the bottleneck
- a real CUDA box (not Home automation)

Until then: `tensor_core: false`, `cuda_graph: false`.
