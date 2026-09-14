# Asahi vs JuniorOS

Asahi is Fedora/Arch on Apple Silicon via m1n1 + linux-asahi. M1/M2 GPU is in-tree-ish; M3 official with weak GPU and no sleep/DCP as of Sep 2026. Neural Engine is out of tree. Pages are 16K.

JuniorOS is not a fork of that kernel. We probe `/etc/os-release` for asahi, then try `import mlx`. Miss → cpu AbsMean. No AGX patch, no ISO.

T1 on an Asahi box: handshake + 32² spine. Metal stays BitNet-mlx on Darwin unless MLX imports here.
