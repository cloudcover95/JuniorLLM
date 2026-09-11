"""Native hook for Climbs/Stock/Home. Loopback only."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from ports.enduser_llm import spec
from ports.inject import digest
from ports.terraform import terraform

HOST = "127.0.0.1"
PORT = 8770


class H(BaseHTTPRequestHandler):
    def log_message(self, *a) -> None:
        return

    def _send(self, code: int, obj: dict) -> None:
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, {"ok": True})
            return
        if path == "/llm":
            self._send(200, spec())
            return
        self._send(404, {"ok": False})

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n).decode() if n else "{}"
        try:
            body = json.loads(raw or "{}")
        except json.JSONDecodeError:
            self._send(400, {"ok": False})
            return
        path = urlparse(self.path).path
        if path == "/terraform":
            self._send(200, terraform(str(body.get("text") or "")))
            return
        if path == "/inject":
            self._send(200, digest(str(body.get("text") or ""), consent=bool(body.get("consent", True)), private=bool(body.get("private", False))))
            return
        self._send(404, {"ok": False})


def serve(host: str = HOST, port: int = PORT) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("loopback only")
    return ThreadingHTTPServer((host, port), H)
