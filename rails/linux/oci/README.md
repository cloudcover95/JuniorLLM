# Rootless OCI (C4)

No docker.sock. No `--privileged`.

```
podman run --user 1000:1000 --read-only --cap-drop=ALL \
  --security-opt no-new-privileges \
  --security-opt seccomp=../seccomp-bitnetd.json \
  --memory 2g --pids-limit 256 --network=none \
  localhost/junior-bitnetd:local
```

See `../CONTAINER_SECURITY.md`.
