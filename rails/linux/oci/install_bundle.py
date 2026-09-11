"""C6 — stage the rootless OCI bundle under an overlay prefix.

Loopback only. No docker.sock. Never fetch weights. Overlay copy, not a kernel.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from rails.linux.oci import rootless

OCI_DIR = Path(__file__).resolve().parent
LINUX = OCI_DIR.parent
BIND = "127.0.0.1:8765"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
FORBIDDEN = ("docker.sock", "0.0.0.0", "/var/run/docker.sock", "/run/docker.sock")

BUNDLE_FILES = (
    (OCI_DIR / "config.json", "usr/lib/junioros/oci/config.json"),
    (OCI_DIR / "README.md", "usr/lib/junioros/oci/README.md"),
    (LINUX / "seccomp-bitnetd.json", "usr/lib/junioros/oci/seccomp-bitnetd.json"),
)


def _bind_ok(bind: str) -> bool:
    host = bind.rsplit(":", 1)[0]
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host in LOOPBACK_HOSTS


def _scan_blob(text: str) -> list[str]:
    issues: list[str] = []
    if "docker.sock" in text:
        issues.append("docker_socket_mentioned")
    if "0.0.0.0" in text:
        issues.append("wildcard:0.0.0.0")
    return issues


def stage(dest: Path | str, bind: str | None = None) -> dict[str, Any]:
    """Copy the C4 OCI bundle into DEST and re-validate the staged config."""
    dest_path = Path(dest)
    bind = bind if bind is not None else os.environ.get("JUNIOR_BIND", BIND)
    issues: list[str] = []
    copied: list[str] = []

    if not str(dest).strip():
        return {
            "ok": False,
            "issues": ["missing_dest"],
            "bind": bind,
            "dest": None,
            "files": [],
            "docker_socket": False,
            "privileged": False,
            "download": False,
        }

    if not _bind_ok(bind):
        return {
            "ok": False,
            "issues": [f"bind_not_loopback:{bind}"],
            "bind": bind,
            "dest": str(dest_path),
            "files": [],
            "docker_socket": False,
            "privileged": False,
            "download": False,
        }

    dest_path.mkdir(parents=True, exist_ok=True)
    for src, rel in BUNDLE_FILES:
        if not src.is_file():
            issues.append(f"missing_source:{src.name}")
            continue
        blob = src.read_text(encoding="utf-8")
        # Policy scan runtime artifacts only (config + seccomp), not docs.
        if src.suffix in {".json"}:
            issues.extend(_scan_blob(blob))
        target = dest_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append(rel)

    staged_cfg = dest_path / "usr/lib/junioros/oci/config.json"
    report: dict[str, Any] = {
        "ok": False,
        "issues": [],
        "bind": BIND,
        "rootless": False,
        "docker_socket": False,
        "privileged": False,
    }
    if staged_cfg.is_file():
        cfg = json.loads(staged_cfg.read_text(encoding="utf-8"))
        report = rootless.validate(cfg)
        issues.extend(report.get("issues") or [])

    # de-dupe while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for item in issues:
        if item not in seen:
            seen.add(item)
            uniq.append(item)

    ok = not uniq and bool(copied) and bool(report.get("ok"))
    return {
        "ok": ok,
        "issues": uniq,
        "bind": report.get("bind") or BIND,
        "dest": str(dest_path),
        "files": copied,
        "rootless": bool(report.get("rootless")),
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "bundle": "usr/lib/junioros/oci",
        "unit": "bitnetd-rootless",
    }


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    dest = args[0] if args else os.environ.get("DEST", "")
    report = stage(dest)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
