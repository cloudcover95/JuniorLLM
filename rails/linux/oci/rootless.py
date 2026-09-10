"""C4 — rootless OCI bitnetd unit. Loopback only. No docker.sock. Never fetch weights."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OCI_DIR = Path(__file__).resolve().parent
LINUX = OCI_DIR.parent
CONFIG_PATH = OCI_DIR / "config.json"
BIND = "127.0.0.1:8765"
FORBIDDEN_MOUNTS = ("docker.sock", "/var/run/docker.sock", "/run/docker.sock")
FORBIDDEN_BINDS = ("0.0.0.0", "::", "[::]")


def load_config(path: Path | None = None) -> dict[str, Any]:
    raw = (path or CONFIG_PATH).read_text(encoding="utf-8")
    return json.loads(raw)


def _env_map(cfg: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in cfg.get("process", {}).get("env", []):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def validate(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg if cfg is not None else load_config()
    issues: list[str] = []
    proc = cfg.get("process") or {}
    linux = cfg.get("linux") or {}
    root = cfg.get("root") or {}
    env = _env_map(cfg)
    bind = env.get("JUNIOR_BIND", "")
    ann = cfg.get("annotations") or {}

    if bind != BIND:
        issues.append(f"bind_not_loopback:{bind or 'missing'}")
    for bad in FORBIDDEN_BINDS:
        blob = json.dumps(cfg)
        if bad in blob:
            issues.append(f"wildcard:{bad}")
            break
    if proc.get("noNewPrivileges") is not True:
        issues.append("missing_noNewPrivileges")
    caps = proc.get("capabilities") or {}
    for key in ("bounding", "effective", "permitted", "ambient"):
        if caps.get(key):
            issues.append(f"caps_{key}")
    if root.get("readonly") is not True:
        issues.append("root_not_readonly")
    nstypes = {ns.get("type") for ns in linux.get("namespaces") or []}
    if "user" not in nstypes:
        issues.append("missing_user_namespace")
    if "network" not in nstypes:
        issues.append("missing_network_namespace")
    mounts = cfg.get("mounts") or []
    for m in mounts:
        src = str(m.get("source") or "") + " " + str(m.get("destination") or "")
        if any(f in src for f in FORBIDDEN_MOUNTS):
            issues.append("docker_socket_mount")
    if "docker.sock" in json.dumps(cfg):
        issues.append("docker_socket_mentioned")
    mem = ((linux.get("resources") or {}).get("memory") or {}).get("limit")
    if not isinstance(mem, int) or mem > 2 * 1024 * 1024 * 1024:
        issues.append("memory_over_2g")
    if ann.get("org.junioros.privileged") != "false":
        issues.append("privileged_annotation")
    if ann.get("org.junioros.docker_socket") != "false":
        issues.append("docker_socket_annotation")

    return {
        "ok": not issues,
        "issues": issues,
        "bind": bind or BIND,
        "rootless": "user" in nstypes,
        "privileged": False,
        "docker_socket": False,
        "readonly": bool(root.get("readonly")),
        "config": str(CONFIG_PATH),
    }


_BITNET_NAMES = ("bitnet-b1.58-2B-4T-I2_S.gguf",)
_CAP_BYTES = 8 * 1024 * 1024 * 1024


def bitnet_weights(root: Path | None = None) -> dict[str, Any]:
    """B2 stale fallback: on-disk path check only. Never download."""
    base = root if root is not None else Path.home() / ".juniorllm" / "models"
    hit = None
    for name in _BITNET_NAMES:
        p = base / name
        try:
            if p.is_file() and p.stat().st_size <= _CAP_BYTES:
                hit = str(p)
                break
        except OSError:
            continue
    return {
        "name": "BitNet-2B4T",
        "present": hit is not None,
        "path": hit,
        "fallback": "JuniorBitNetFieldCore",
        "download": False,
    }


def unit() -> dict[str, Any]:
    """Declarative rootless unit the overlay can render to systemd --user later."""
    check = validate()
    weights = bitnet_weights()
    return {
        "name": "bitnetd-rootless",
        "kind": "oci-rootless",
        "bind": BIND,
        "exec": ["/usr/local/bin/junior-bitnetd"],
        "user": "1000:1000",
        "read_only": True,
        "cap_drop": "ALL",
        "no_new_privileges": True,
        "memory": "2g",
        "pids": 256,
        "network": "none",
        "seccomp": str(LINUX / "seccomp-bitnetd.json"),
        "docker_socket": False,
        "privileged": False,
        "oci_ok": check["ok"],
        "weights": weights,
        "status": "ready" if check["ok"] else "invalid",
    }


def main() -> int:
    report = {"validate": validate(), "unit": unit()}
    print(json.dumps(report, indent=2))
    return 0 if report["validate"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
