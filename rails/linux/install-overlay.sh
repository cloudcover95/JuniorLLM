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
  "$DEST/usr/lib" \
  "$DEST/usr/local/bin"

cp "$HERE/os-release.junior" "$DEST/etc/os-release.junior"
cp "$HERE/os-release.junior" "$DEST/usr/lib/os-release.junior"
cp "$HERE/bitnetd.service" "$DEST/etc/systemd/system/bitnetd.service"
cp "$HERE/bitnetd.sh" "$DEST/usr/local/bin/junior-bitnetd"
cp "$HERE/juniorctl.py" "$DEST/usr/local/bin/juniorctl"
chmod 0755 "$DEST/usr/local/bin/junior-bitnetd" "$DEST/usr/local/bin/juniorctl"

# Fail closed if a copied unit advertises a wildcard bind.
if grep -q '0.0.0.0' "$DEST/etc/systemd/system/bitnetd.service"; then
  echo "refusing unit with wildcard bind" >&2
  exit 3
fi

echo "installed JuniorOS overlay under $DEST bind=$BIND"
exit 0
