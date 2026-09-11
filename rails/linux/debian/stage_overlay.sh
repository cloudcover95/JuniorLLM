#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
OUT=${1:-/tmp/junioros-overlay}
export DEST="$OUT"
sh "$HERE/install-overlay.sh"
cp "$HERE/kconfig.junior" "$OUT/usr/lib/junioros/kconfig.junior"
cp "$HERE/debian/WHY_NOT_VMLINUZ.md" "$OUT/usr/lib/junioros/WHY_NOT_VMLINUZ.md"
cp "$HERE/debian/packages.list" "$OUT/usr/lib/junioros/packages.list"
cp "$HERE/i2sd.py" "$HERE/backend.py" "$HERE/bitnet_cpp.py" "$OUT/usr/lib/junioros/"
echo "staged overlay files; run from a JuniorLLM tree so PYTHONPATH finds junior_bitnet"
tar -C "$OUT" -czf "$OUT.tar.gz" .
echo "staged $OUT and $OUT.tar.gz"
