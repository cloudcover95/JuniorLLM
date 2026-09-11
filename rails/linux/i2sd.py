"""I2_S daemon. Loopback only. Vendor kernel."""
from __future__ import annotations

import socketserver
import threading

from junior_bitnet.i2s import pack_floats, unpack

HOST = "127.0.0.1"
PORT = 8767


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        if self.client_address[0] not in {"127.0.0.1", "::1"}:
            return
        line = self.rfile.readline().decode("utf-8", "replace").strip()
        if line == "PING":
            self.wfile.write(b"PONG\n")
            return
        if line.startswith("PACK "):
            xs = [float(x) for x in line[5:].split(",") if x.strip()]
            blob, n, scale = pack_floats(xs)
            self.wfile.write(f"OK {n} {scale:.6f} {blob.hex()}\n".encode())
            return
        if line.startswith("UNPACK "):
            _, hexb, n_s = line.split()
            z = unpack(bytes.fromhex(hexb), int(n_s))
            self.wfile.write(("OK " + ",".join(str(t) for t in z) + "\n").encode())
            return
        self.wfile.write(b"ERR\n")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def serve(host: str = HOST, port: int = PORT) -> Server:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("loopback only")
    return Server((host, port), Handler)


def start_thread(port: int = PORT) -> tuple[Server, threading.Thread]:
    srv = serve("127.0.0.1", port)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, t


if __name__ == "__main__":
    print(f"i2sd {HOST}:{PORT}")
    serve().serve_forever()
