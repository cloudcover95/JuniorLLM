#!/bin/sh
# Placeholder BitNet Linux OS rail. Replace with bitnet.cpp server later.
# Binds only if JUNIOR_BIND is loopback.
BIND=${JUNIOR_BIND:-127.0.0.1:8765}
echo "junior-bitnetd offline=$JUNIOR_OFFLINE bind=$BIND"
exec python3 -m http.server --bind 127.0.0.1 8765
