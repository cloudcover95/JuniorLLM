"""C7 — pin juniorctl on overlay PATH and stage a systemd --user unit.

Loopback only. Overlay copy, not a kernel. No docker.sock. Never fetch weights.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

LINUX = Path(__file__).resolve().parent
BIND = "127.0.0.1:8765"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
BIN_DIR = "/usr/local/bin"
PROFILE_REL = "etc/profile.d/junioros.sh"
USER_UNIT_REL = "usr/lib/systemd/user/bitnetd.service"
PATH_ENV_REL = "etc/junioros/path.env"

PROFILE_SH = """# JuniorOS overlay PATH pin — loopback bitnetd only
# Installed under DEST/etc/profile.d/junioros.sh
JUNIOROS_BIN="/usr/local/bin"
case ":$PATH:" in
  *":$JUNIOROS_BIN:"*) ;;
  *) PATH="$JUNIOROS_BIN:$PATH" ;;
esac
export PATH
export JUNIOR_BIND="${JUNIOR_BIND:-127.0.0.1:8765}"
"""

USER_UNIT = """[Unit]
Description=Junior BitNet local inference (user, loopback)
After=default.target

[Service]
Type=simple
ExecStart=/usr/local/bin/junior-bitnetd
Restart=on-failure
Environment=JUNIOR_OFFLINE=1
Environment=JUNIOR_BIND=127.0.0.1:8765
NoNewPrivileges=yes
ProtectSystem=strict
PrivateTmp=yes
MemoryDenyWriteExecute=yes
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
RestrictNamespaces=yes
CapabilityBoundingSet=
AmbientCapabilities=
MemoryMax=2G
TasksMax=256

[Install]
WantedBy=default.target
"""

PATH_ENV = """JUNIOROS_BIN=/usr/local/bin
JUNIOR_BIND=127.0.0.1:8765
JUNIOR_USER_UNIT=usr/lib/systemd/user/bitnetd.service
"""


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
    """Write PATH pin + systemd user unit under DEST."""
    dest_path = Path(dest)
    bind = bind if bind is not None else os.environ.get("JUNIOR_BIND", BIND)
    issues: list[str] = []
    files: list[str] = []

    if not str(dest).strip():
        return {
            "ok": False,
            "issues": ["missing_dest"],
            "bind": bind,
            "dest": None,
            "files": [],
            "path_pin": False,
            "user_unit": False,
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
            "path_pin": False,
            "user_unit": False,
            "docker_socket": False,
            "privileged": False,
            "download": False,
        }

    blobs = (
        (PROFILE_REL, PROFILE_SH),
        (USER_UNIT_REL, USER_UNIT),
        (PATH_ENV_REL, PATH_ENV),
    )
    dest_path.mkdir(parents=True, exist_ok=True)
    for rel, blob in blobs:
        issues.extend(_scan_blob(blob))
        target = dest_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(blob, encoding="utf-8")
        if rel.endswith(".sh"):
            target.chmod(0o644)
        files.append(rel)

    unit_text = (dest_path / USER_UNIT_REL).read_text(encoding="utf-8")
    profile_text = (dest_path / PROFILE_REL).read_text(encoding="utf-8")
    if BIND not in unit_text or BIND not in profile_text:
        issues.append("bind_missing")
    if "WantedBy=default.target" not in unit_text:
        issues.append("user_unit_not_default_target")
    if BIN_DIR not in profile_text:
        issues.append("path_bin_missing")
    if "User=root" in unit_text or "User=nobody" in unit_text:
        issues.append("user_unit_has_system_user")

    seen: set[str] = set()
    uniq: list[str] = []
    for item in issues:
        if item not in seen:
            seen.add(item)
            uniq.append(item)

    ok = not uniq and len(files) == 3
    return {
        "ok": ok,
        "issues": uniq,
        "bind": BIND,
        "dest": str(dest_path),
        "files": files,
        "path_pin": PROFILE_REL,
        "user_unit": USER_UNIT_REL,
        "bin": BIN_DIR,
        "docker_socket": False,
        "privileged": False,
        "download": False,
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
