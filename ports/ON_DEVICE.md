# On-device loaders (no fetch)

Automation and `juniorctl` must **not** download weights.

| Port | File if present | Else |
|------|-----------------|------|
| BitNet-2B4T | `~/.juniorllm/models/bitnet-b1.58-2B-4T-I2_S.gguf` | stay on FieldCore ternary |
| JuniorGemma4-4B | `~/.juniorllm/models/gemma4-4b-q4_k_m.gguf` or MLX dir | interactive stub in `adaptations/gemma4` |
| Qwen-local | `~/.juniorllm/models/qwen3-8b-q4_k_m.gguf` | Gemma or FieldCore |
| JuniorKimiK3-edge | never pull 1.5TB | pruned stub only |

`ports.registry.pick` already caps RAM. Loaders check path existence before mlx/gguf.
