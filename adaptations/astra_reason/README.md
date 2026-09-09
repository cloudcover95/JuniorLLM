# JuniorAstraReason

**Weight stand-in for Astra-class reasoning** — separate from `JuniorAstra` (the open runtime).

Closed GPT-6 Astra has no local weights. This port uses:

1. **High-quant base family** — Gemma 4 4B Q4_K_M / MLX *or* Qwen-local Q4_K_M / GGUF (cap 8 GB).
2. **BitNet rigid IQ layers** — looped ternary stack (`{-1,0,1}`). Extra loops = extra rigidity, same parameter count (looped-transformer idea, implemented locally).
3. **MemSys palace** — bit-drift notes from JuniorMemSys-Suite contract, stdlib in-tree so the van does not need that repo cloned.

Honest limit: this does **not** match frontier Astra scores. It is the local compensation stack (memory + rigidity + high-quant base).

## Files
- `config.py` — base family + loop depth + RAM cap
- `rigid_iq.py` — looped ternary IQ layers
- `memsys_bridge.py` — palace put/get
- `stack.py` — reason() entry
