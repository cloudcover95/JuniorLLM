# Roadmap — custom BitNet Linux OS (JuniorOS)

Not a from-scratch kernel. An **overlay** on a minimal Debian/Alpine userspace that treats BitNet as PID 1-adjacent inference.

Full night charter: `docs/BETA_TO_OS.md`.

## Layers

| Layer | Now | Next |
|-------|-----|------|
| 0 Hardware | M4 / Pi / any x86 8 GB+ | same |
| 1 Userspace | stock Linux | Alpine or Debian netinst |
| 2 junior-bitnetd | `rails/linux/bitnetd.service` loopback placeholder | real bitnet.cpp I2_S 2B4T if GGUF on disk |
| 3 JuniorLLM ports | FieldCore Gemma Fable Astra AstraReason Kimi-edge Qwen Night | on-device loaders only |
| 4 Local apps | JuniorClimbs beta + `juniorctl ask` | `juniorctl` on PATH |
| 5 Policy | Guardrail + covenant + Fable | OS-level seccomp later |
| 6 Overnight | DreamMesh + Grok Automations | systemd timer → local cycle first |
| 7 AIE | `junior_aie` 15-piece | keep stdlib |

## juniorctl

```
juniorctl health
juniorctl port list
juniorctl ask "flagstaff late summer?"
juniorctl night
```

Do not claim an OSWorld score. Do not ship a custom kernel blob.
