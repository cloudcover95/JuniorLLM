"""Loopback Home page. Form → user_in.ingest."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from ports.user_in import ingest

HOST = "127.0.0.1"
PORT = 8771
PAGE = b"""<!doctype html><meta charset=utf-8><title>JuniorHome</title>
<body style="font-family:sans-serif;max-width:40rem;margin:2rem">
<h1>JuniorHome</h1>
<p>Local only. Any note. Guess routes CAD/app/media or user.</p>
<form method=post>
<textarea name=note rows=6 cols=60 required></textarea><br>
<button>terraform</button>
</form>
</body>"""


class H(BaseHTTPRequestHandler):
    vault = Path("/tmp/juniorhome_vault")

    def log_message(self, *a) -> None:
        return

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE)

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        body = parse_qs(self.rfile.read(n).decode())
        note = (body.get("note") or [""])[0]
        out = ingest(self.vault, note)
        msg = f"ok={out['ok']} area={out['guess']} inbox={out.get('inbox')}\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(msg)))
        self.end_headers()
        self.wfile.write(msg)


def serve(vault: Path, host: str = HOST, port: int = PORT) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("loopback only")
    H.vault = Path(vault)
    return ThreadingHTTPServer((host, port), H)
