from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junior_bitnet.i2s import pack, unpack
from rails.linux.i2sd import start_thread


class I2STests(unittest.TestCase):
    def test_roundtrip(self):
        z = [1, 0, -1, 1, 0]
        self.assertEqual(unpack(pack(z), len(z)), z)

    def test_daemon(self):
        srv, _ = start_thread(18767)
        try:
            with socket.create_connection(("127.0.0.1", 18767), 2) as s:
                s.sendall(b"PING\n")
                self.assertEqual(s.recv(16).decode(), "PONG\n")
            with socket.create_connection(("127.0.0.1", 18767), 2) as s:
                s.sendall(b"PACK 40,20,8,0.2,-3\n")
                line = s.recv(256).decode().strip().split()
                self.assertEqual(line[0], "OK")
                n = int(line[1])
                blob = bytes.fromhex(line[3])
                z = unpack(blob, n)
                self.assertTrue(all(t in (-1, 0, 1) for t in z))
        finally:
            srv.shutdown()
            srv.server_close()

    def test_refuse_wildcard(self):
        from rails.linux.i2sd import serve

        with self.assertRaises(ValueError):
            serve("0.0.0.0", 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
