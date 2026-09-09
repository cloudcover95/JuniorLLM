#!/bin/sh
# JuniorOS overlay installer — copy identity + bitnetd unit. Loopback only.
# Does not replace host /etc/os-release. Does not enable systemd unless --enable.
# Usage: install-overlay.sh [--dest DIR] [--dry-run] [--enable]
set -eu

HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
DEST=${JUNIOR_OVERLAY_ROOT:-}
BIND=${JUNIOR_BIND:-127.0.0.1:8765}
DRY=0
ENABLE=0

usage() {
  echo "usage: install-overlay.sh [--dest DIR] [--dry-run] [--enable]" >&2
  echo "  JUNIOR_OVERLAY_ROOT or --dest required (prefix, not live / unless intended)" >&2
  echo "  JUNIOR_BIND must be 127.0.0.1:port (default 127.0.0.1:8765)" >&2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dest)
      DEST=${2:-}
      shift 2
      ;;
    --dry-run)
      DRY=1
      shift
      ;;
    --enable)
      ENABLE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [ -z "$DEST" ]; then
  usage
  exit 2
fi

case "$BIND" in
  127.0.0.1:*)
    :
    ;;
  *)
    echo "refusing non-loopback JUNIOR_BIND=$BIND" >&2
    exit 3
    ;;
esac

UNIT_SRC="$HERE/bitnetd.service"
REL_SRC="$HERE/os-release.junior"
DAEMON_SRC="$HERE/bitnetd.sh"
CTL_SRC="$HERE/juniorctl.py"

for f in "$UNIT_SRC" "$REL_SRC" "$DAEMON_SRC" "$CTL_SRC"; do
  if [ ! -f "$f" ]; then
    echo "missing source $f" >&2
    exit 4
  fi
done

if grep -E '0\.0\.0\.0' "$UNIT_SRC" "$DAEMON_SRC" "$REL_SRC" >/dev/null 2>&1; then
  echo "refusing source that advertises 0.0.0.0" >&2
  exit 3
fi

UNIT_DST="$DEST/etc/systemd/system/bitnetd.service"
REL_DST="$DEST/etc/os-release.junior"
DAEMON_DST="$DEST/usr/local/bin/junior-bitnetd"
CTL_DST="$DEST/usr/local/bin/juniorctl"

echo "JuniorOS overlay dest=$DEST bind=$BIND dry=$DRY enable=$ENABLE"

if [ "$DRY" -eq 1 ]; then
  echo "would copy $REL_SRC -> $REL_DST"
  echo "would copy $UNIT_SRC -> $UNIT_DST"
  echo "would copy $DAEMON_SRC -> $DAEMON_DST"
  echo "would copy $CTL_SRC -> $CTL_DST"
  exit 0
fi

mkdir -p "$DEST/etc/systemd/system" "$DEST/usr/local/bin"
cp "$REL_SRC" "$REL_DST"
cp "$UNIT_SRC" "$UNIT_DST"
cp "$DAEMON_SRC" "$DAEMON_DST"
cp "$CTL_SRC" "$CTL_DST"
chmod 0755 "$DAEMON_DST" "$CTL_DST"

# Keep advertised bind loopback even if the unit file is later edited on disk.
if ! grep -q '127.0.0.1:8765' "$UNIT_DST"; then
  echo "unit missing loopback bind after copy" >&2
  exit 3
fi

if [ "$ENABLE" -eq 1 ]; then
  if command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
    echo "enable skipped: this slice copies units only; systemctl enable is a later slice"
  else
    echo "enable skipped: no systemd runtime"
  fi
fi

echo "installed overlay identity + bitnetd unit (loopback)"
echo "$REL_DST"
echo "$UNIT_DST"
echo "$DAEMON_DST"
echo "$CTL_DST"
