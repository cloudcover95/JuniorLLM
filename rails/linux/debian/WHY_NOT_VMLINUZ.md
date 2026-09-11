# Why JuniorOS is not a custom vmlinuz

A unique kernel blob breaks the three boxes this stack actually runs on:

- **MLX** is userspace on Darwin / Asahi. It does not load a Junior `vmlinuz`.
- **CUDA** is tied to the distro + NVIDIA ABI. A forked kernel is how you lose the driver.
- **ARM** (Pi, Ampere, van SBC) ships a vendor kernel. Replace it and you lose DTBs and modules.

Ternary optimization that *is* kernel-adjacent:

1. Stock Debian/Alpine/`raspberrypi-kernel` + this overlay (`install-overlay.sh`).
2. kconfig *wants* already on those kernels: namespaces, seccomp, cgroups v2.
3. Same AbsMean in `rails/linux/backend.py` on mlx / cuda / cpu.
4. Optional later: bitnet.cpp as a userspace daemon, not a syscall.

If an ISO is needed, it is **Debian netinst + late_command** that copies this overlay. Not a rebuilt `linux-image`.
