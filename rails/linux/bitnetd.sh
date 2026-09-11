#!/bin/sh
# Loopback only. Exec bitnet.cpp when JUNIOR_BITNET_CPP + JUNIOR_GGUF exist.
set -eu
BIND=${JUNIOR_BIND:-127.0.0.1:8765}
host=${BIND%:*}
case "$host" in
  127.0.0.1|localhost|::1) ;;
  *) echo "refusing non-loopback $BIND" >&2; exit 3 ;;
esac
if [ -n "${JUNIOR_BITNET_CPP:-}" ] && [ -x "$JUNIOR_BITNET_CPP" ] && [ -n "${JUNIOR_GGUF:-}" ] && [ -f "$JUNIOR_GGUF" ]; then
  echo "junior-bitnetd cpp=$JUNIOR_BITNET_CPP bind=$BIND"
  exec "$JUNIOR_BITNET_CPP" -m "$JUNIOR_GGUF" --host 127.0.0.1 --port 8765
fi
echo "junior-bitnetd placeholder bind=$BIND"
exec python3 -m http.server --bind 127.0.0.1 8765
