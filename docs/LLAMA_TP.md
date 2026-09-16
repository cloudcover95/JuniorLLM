# llama.cpp tensor parallelism backlog

TP splits layers across GPUs (`--tensor-split` / pipeline).
Home T3 reads a local GGUF header only. No multi-GPU serve from automation.

Backlog when all true:
- local GGUF on disk
- llama-cli present
- two devices
- bench_pipe says host pack is not the bottleneck

Until then TP = false.
