from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rails.linux.i2sd import start_thread
from rails.linux.iot import pack_iot, pack_local


class IotTests(unittest.TestCase):
    def test_local_and_daemon(self):
        xs = [1.2, -0.4, 0.05]
        loc = pack_local(xs)
        self.assertEqual(loc["via"], "local")
        srv, _ = start_thread(18767)
        try:
            d = pack_iot(xs, port=18767)
            self.assertEqual(d["via"], "i2sd")
            self.assertEqual(d["i2s"], loc["i2s"])
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
