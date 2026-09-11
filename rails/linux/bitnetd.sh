#!/bin/sh
set -eu
BIND=${JUNIOR_BIND:-127.0.0.1:8765}
host=${BIND%:*}
case "$host" in
  127.0.0.1|localhost|::1) ;;
  *) echo "refusing non-loopback $BIND" >&2; exit 3 ;;
esac
if [ -n "${JUNIOR_BITNET_CPP:-}" ] && [ -x "$JUNIOR_BITNET_CPP" ] && [ -n "${JUNIOR_GGUF:-}" ] && [ -f "$JUNIOR_GGUF" ]; then
  exec "$JUNIOR_BITNET_CPP" -m "$JUNIOR_GGUF" --host 127.0.0.1 --port 8765
fi
HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
export PYTHONPATH=${PYTHONPATH:-$(CDPATH= cd -- "$HERE/../.." && pwd)}
exec python3 "$HERE/i2sd.py"
