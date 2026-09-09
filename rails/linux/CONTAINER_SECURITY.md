# JuniorOS + Linux container security

This is host and workload isolation for the custom OS overlay.
It is not a travel or edge-lifestyle document.

Containers share the host kernel. Isolation is namespaces + cgroups + capabilities + seccomp (+ LSM). A breakout is a kernel or runtime bug away from host root unless user namespaces and a reduced syscall surface are in play.

## Threat model (JuniorOS)

- `bitnetd` is local inference. It must not be reachable off loopback.
- Model files and the AIE sandbox are untrusted compute adjacent to the host.
- Compromise of a container must not yield host root, `CAP_SYS_ADMIN`, or the Docker/podman socket.
- Unprivileged user namespaces help rootless runtimes and hurt some LPE chains if left wide open. JuniorOS does **not** globally set `user.max_user_namespaces=0` if rootless OCI is required. Hosts that never run rootless may lock that knob.

## Controls we ship in-tree

| Control | File |
|---------|------|
| systemd sandbox for bitnetd | `bitnetd.service` |
| seccomp profile (OCI / runc) | `seccomp-bitnetd.json` |
| host sysctl notes | `sysctl-junior.conf` |
| overlay installer | `install-overlay.sh` |
| policy check | `juniorctl security` |

## Required runtime flags (OCI / Docker / Podman)

```
--read-only
--cap-drop=ALL
--security-opt=no-new-privileges
--security-opt seccomp=seccomp-bitnetd.json
--memory=2g --pids-limit=256
--user 1000:1000
--network=none
```

If HTTP loopback is required, use a userspace slirp/pasta net and still bind `127.0.0.1` inside. Never `--privileged`. Never mount `/var/run/docker.sock`.

Drop `CAP_NET_RAW`, `CAP_SYS_ADMIN`, `CAP_SYS_PTRACE`, `CAP_SYS_MODULE`.

## Layers beyond runc (optional, stronger)

- **gVisor / Kata** — extra kernel or VM boundary when inference is fed untrusted prompts at scale.
- **Rootless** kubelet / CRI (KubeletInUserNamespace beta in k8s 1.37) if this overlay ever sits under a node agent.
- Image pin by digest + signature (cosign). No `:latest`.

## What this does not do

Does not patch the kernel. Does not claim a hardened distro. Does not disable user namespaces on developer images that need rootless Podman.
