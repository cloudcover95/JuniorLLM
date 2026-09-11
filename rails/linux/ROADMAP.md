# Roadmap — BitNet Linux OS (JuniorOS)

Overlay on vendor kernels. Not a custom vmlinuz. See `debian/WHY_NOT_VMLINUZ.md`.

| Layer | Now |
|-------|-----|
| 0 Hardware | backend.py mlx / cuda / cpu |
| 1 Userspace | stock Debian/Alpine/Pi kernel |
| 1b Overlay | `debian/stage_overlay.sh` → tar.gz |
| 2 bitnetd | loopback unit |
| 3 Ports | layer_mgr + AbsMean |

```
sh rails/linux/debian/stage_overlay.sh /tmp/junioros-overlay
PYTHONPATH=. python scripts/os_probe.py
```
