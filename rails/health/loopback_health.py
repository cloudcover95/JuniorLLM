"""JuniorLLM rails: loopback-only health endpoint.

Stdlib-only, local-first health probe for JuniorLLM rails. It binds to
127.0.0.1 only (refuses any other host), pulls no model, and needs no CI.

Each /health response carries a close hash that chains to the previous one:
    close = sha256(prev || pack || text)[:16]
where `pack` is the rail's trit pack tag and the trit rate is 0.05.

Run:  python rails/health/loopback_health.py --selftest
      python rails/health/loopback_health.py --port 8779
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOOPBACK = "127.0.0.1"
TRIT_RATE = 0.05
PACK = "juniorllm-rails-health"
GENESIS = "0" * 16


def close_hash(prev: str, pack: str, text: str) -> str:
    """Close rule: sha256(prev || pack || text)[:16]."""
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


class HealthState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prev = GENESIS
        self.started = time.time()
        self.count = 0

    def tick(self) -> dict:
        with self._lock:
            self.count += 1
            text = f"ok:{self.count}"
            close = close_hash(self.prev, PACK, text)
            body = {
                "status": "ok",
                "seq": self.count,
                "uptime_s": round(time.time() - self.started, 3),
                "pack": PACK,
                "trit_rate": TRIT_RATE,
                "prev": self.prev,
                "close": close,
                "bind": LOOPBACK,
            }
            self.prev = close
            return body


def make_handler(state: HealthState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_error(404)
                return
            data = json.dumps(state.tick()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:
            return

    return Handler


def serve(port: int, host: str = LOOPBACK) -> ThreadingHTTPServer:
    if host != LOOPBACK:
        raise ValueError(f"refusing to bind {host!r}; loopback {LOOPBACK} only")
    return ThreadingHTTPServer((host, port), make_handler(HealthState()))


def selftest() -> None:
    # close rule is deterministic and 16 hex chars
    a = close_hash(GENESIS, PACK, "ok:1")
    assert len(a) == 16 and a == close_hash(GENESIS, PACK, "ok:1")
    # non-loopback bind is refused
    for bad in ("0.0.0.0", "::", "192.168.1.10"):
        try:
            serve(0, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bound {bad}")
    # live round-trip on an ephemeral loopback port, chain links correctly
    srv = serve(0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://{LOOPBACK}:{srv.server_address[1]}/health"
        r1 = json.loads(urllib.request.urlopen(url, timeout=5).read())
        r2 = json.loads(urllib.request.urlopen(url, timeout=5).read())
        assert r1["prev"] == GENESIS and r1["close"] == a
        assert r2["prev"] == r1["close"]
        assert r2["close"] == close_hash(r1["close"], PACK, "ok:2")
        assert r1["trit_rate"] == TRIT_RATE and r1["bind"] == LOOPBACK
    finally:
        srv.shutdown()
        srv.server_close()
    print("selftest ok")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8779)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    srv = serve(args.port)
    print(f"health on http://{LOOPBACK}:{args.port}/health")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
