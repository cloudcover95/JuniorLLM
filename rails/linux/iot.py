"""Tiny IoT client for I2_S. Loopback daemon if up, else local pack."""
from __future__ import annotations

import socket

from junior_bitnet.i2s import pack_floats


def pack_local(xs: list[float]) -> dict:
    blob, n, scale = pack_floats(xs)
    return {"via": "local", "n": n, "scale": scale, "i2s": blob.hex()}


def pack_iot(xs: list[float], host: str = "127.0.0.1", port: int = 8767, timeout: float = 0.2) -> dict:
    line = "PACK " + ",".join(str(x) for x in xs) + "\n"
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.sendall(line.encode())
            raw = s.recv(4096).decode()
        parts = raw.split()
        if parts[:1] != ["OK"]:
            return pack_local(xs) | {"via": "local-fallback"}
        return {"via": "i2sd", "n": int(parts[1]), "scale": float(parts[2]), "i2s": parts[3]}
    except OSError:
        return pack_local(xs)
