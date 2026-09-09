#!/bin/sh
# Install JuniorOS overlay units. Requires root. Does not touch the kernel.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
PREFIX=${PREFIX:-/usr/local}
UNIT_DIR=${UNIT_DIR:-/etc/systemd/system}
SYSCTL_DIR=${SYSCTL_DIR:-/etc/sysctl.d}

install -d "$PREFIX/bin" "$PREFIX/share/junioros"
install -m 0755 "$ROOT/bitnetd.sh" "$PREFIX/bin/junior-bitnetd"
install -m 0755 "$ROOT/juniorctl.py" "$PREFIX/bin/juniorctl"
install -m 0644 "$ROOT/seccomp-bitnetd.json" "$PREFIX/share/junioros/seccomp-bitnetd.json"
install -m 0644 "$ROOT/CONTAINER_SECURITY.md" "$PREFIX/share/junioros/CONTAINER_SECURITY.md"

if [ "$(id -u)" -eq 0 ]; then
  install -m 0644 "$ROOT/bitnetd.service" "$UNIT_DIR/bitnetd.service"
  install -m 0644 "$ROOT/sysctl-junior.conf" "$SYSCTL_DIR/90-junior.conf"
  echo "installed units; systemctl daemon-reload && systemctl enable --now bitnetd"
else
  echo "non-root: files copied under $PREFIX only"
fi
