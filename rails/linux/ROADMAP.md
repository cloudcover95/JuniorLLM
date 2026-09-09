# Roadmap — custom BitNet Linux OS (JuniorOS)

Not a from-scratch kernel. An **overlay** on a minimal Debian/Alpine userspace that treats BitNet as PID 1-adjacent inference.

## Layers

| Layer | Now | Next |
|-------|-----|------|
| 0 Hardware | M4 / Pi / any x86 8 GB+ | same |
| 1 Userspace | stock Linux | Alpine or Debian netinst |
| 2 junior-bitnetd | `rails/linux/bitnetd.service` loopback placeholder | real bitnet.cpp I2_S 2B4T |
| 3 JuniorLLM ports | Gemma / Fable / Astra / FieldCore / Night | loaders on device |
| 4 Local apps | JuniorClimbs 0.9.3-beta, JuniorHome scripts | single `juniorctl` |
| 5 Policy | Guardrail + covenant + Fable | OS-level seccomp later |
| 6 Overnight | DreamMesh + Grok Automations | systemd timer calls local cycle first, Grok Bot only for git |

## juniorctl (target)

```
juniorctl health
juniorctl night --ticks 64
juniorctl port list
juniorctl stonefield serve
```

## What Grok Bot should add next (files)

- `rails/linux/juniorctl.py` — stdlib CLI stub
- `rails/linux/os-release.junior` — NAME=JuniorOS
- `rails/linux/sysctl-junior.conf` — notes only
- Keep `bitnetd` on 127.0.0.1:8765

Do not claim an OSWorld score. Do not ship a custom kernel blob.
