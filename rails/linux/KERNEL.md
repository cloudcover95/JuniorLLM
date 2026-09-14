# JuniorOS kernel architecture

Not a vmlinuz we ship.

```
host kernel (vendor / asahi / debian)
    └ userspace overlay
         ├ i2sd          loopback trit daemon
         ├ juniorctl     bind 127.0.0.1 only
         ├ os_route      Home T0–T2
         └ bitnet.cpp    probe if binary on disk
```

`kconfig.junior` is a *wanted* fragment (loopback, no unprivileged bpf). It is not applied by automation.
`os-release.junior` is identity text for an overlay, not /etc on this VM.
1.58 lives in userspace (`junior_bitnet.winsor`). No trit opcode in the ISA.
