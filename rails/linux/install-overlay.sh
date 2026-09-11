#!/bin/sh
# JuniorOS overlay installer — copies os-release + bitnetd unit + wrappers into DEST.
# Overlay only. Loopback bind only. No kernel blob. No 0.0.0.0.
set -eu

HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
DEST=${DEST:-}
BIND=${JUNIOR_BIND:-127.0.0.1:8765}

usage() {
  echo "usage: DEST=/opt/junioros $0" >&2
  echo "  copies os-release.junior, bitnetd.service, junior-bitnetd, juniorctl under DEST" >&2
  echo "  JUNIOR_BIND must be loopback (default 127.0.0.1:8765)" >&2
  exit 2
}

host=${BIND%:*}
case "$host" in
  127.0.0.1|localhost|::1)
    ;;
  *)
    echo "refusing non-loopback JUNIOR_BIND=$BIND" >&2
    exit 3
    ;;
esac

[ -n "$DEST" ] || usage

mkdir -p \
  "$DEST/etc/systemd/system" \
  "$DEST/etc/profile.d" \
  "$DEST/etc/junioros" \
  "$DEST/usr/lib" \
  "$DEST/usr/lib/junioros/oci" \
  "$DEST/usr/lib/systemd/user" \
  "$DEST/usr/local/bin"

cp "$HERE/os-release.junior" "$DEST/etc/os-release.junior"
cp "$HERE/os-release.junior" "$DEST/usr/lib/os-release.junior"
cp "$HERE/bitnetd.service" "$DEST/etc/systemd/system/bitnetd.service"
cp "$HERE/bitnetd.sh" "$DEST/usr/local/bin/junior-bitnetd"
cp "$HERE/juniorctl.py" "$DEST/usr/local/bin/juniorctl"
# C6 — stage the C4 rootless OCI bundle next to the overlay units.
cp "$HERE/oci/config.json" "$DEST/usr/lib/junioros/oci/config.json"
cp "$HERE/oci/README.md" "$DEST/usr/lib/junioros/oci/README.md"
cp "$HERE/seccomp-bitnetd.json" "$DEST/usr/lib/junioros/oci/seccomp-bitnetd.json"
chmod 0755 "$DEST/usr/local/bin/junior-bitnetd" "$DEST/usr/local/bin/juniorctl"

# C7 — PATH pin + systemd --user unit (loopback only).
cat > "$DEST/etc/profile.d/junioros.sh" <<'EOF'
# JuniorOS overlay PATH pin — loopback bitnetd only
# Installed under DEST/etc/profile.d/junioros.sh
JUNIOROS_BIN="/usr/local/bin"
case ":$PATH:" in
  *":$JUNIOROS_BIN:"*) ;;
  *) PATH="$JUNIOROS_BIN:$PATH" ;;
esac
export PATH
export JUNIOR_BIND="${JUNIOR_BIND:-127.0.0.1:8765}"
EOF
cat > "$DEST/usr/lib/systemd/user/bitnetd.service" <<'EOF'
[Unit]
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
EOF
cat > "$DEST/etc/junioros/path.env" <<'EOF'
JUNIOROS_BIN=/usr/local/bin
JUNIOR_BIND=127.0.0.1:8765
JUNIOR_USER_UNIT=usr/lib/systemd/user/bitnetd.service
EOF

# Fail closed if a copied unit or OCI config advertises a wildcard bind or docker.sock.
if grep -Fq '0.0.0.0' "$DEST/etc/systemd/system/bitnetd.service" \
  "$DEST/usr/lib/systemd/user/bitnetd.service" \
  "$DEST/etc/profile.d/junioros.sh" \
  "$DEST/usr/lib/junioros/oci/config.json"; then
  echo "refusing unit with wildcard bind" >&2
  exit 3
fi
if grep -Fq 'docker.sock' "$DEST/usr/lib/junioros/oci/config.json"; then
  echo "refusing OCI bundle that mentions docker.sock" >&2
  exit 3
fi

echo "installed JuniorOS overlay under $DEST bind=$BIND"
exit 0
