#!/bin/sh
# Stage overlay tarball. Does not build an ISO or a kernel.
set -eu
HERE=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
OUT=${1:-/tmp/junioros-overlay}
export DEST="$OUT"
sh "$HERE/install-overlay.sh"
cp "$HERE/kconfig.junior" "$OUT/usr/lib/junioros/kconfig.junior"
cp "$HERE/debian/WHY_NOT_VMLINUZ.md" "$OUT/usr/lib/junioros/WHY_NOT_VMLINUZ.md"
cp "$HERE/debian/packages.list" "$OUT/usr/lib/junioros/packages.list"
tar -C "$OUT" -czf "$OUT.tar.gz" .
echo "staged $OUT and $OUT.tar.gz"
