# Roadmap — custom BitNet Linux OS (JuniorOS)

Not a from-scratch kernel. Overlay on Debian/Alpine. BitNet is inference, not PID 1.

## Layers

| Layer | Now | Next |
|-------|-----|------|
| 0 Hardware | `rails/linux/backend.py` probe: mlx / cuda / cpu ARM+x86 | same AbsMean on all |
| 1 Userspace | stock Linux / Darwin | Alpine or Debian netinst |
| 2 junior-bitnetd | bitnetd.service loopback | bitnet.cpp I2_S if GGUF on disk |
| 3 Ports | FieldCore + layer_mgr phase | on-device loaders only |
| 4 Apps | juniorctl | PATH |
| 5 Policy | Guardrail + Fable | seccomp later |
| 6 Night | DreamMesh | local timer first |

```
PYTHONPATH=. python scripts/os_probe.py
```

Do not claim an OSWorld score. Do not ship a custom kernel blob.
