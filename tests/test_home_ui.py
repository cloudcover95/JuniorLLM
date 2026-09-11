from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rails.linux.home_ui import serve


class UiTests(unittest.TestCase):
    def test_post(self):
        with tempfile.TemporaryDirectory() as td:
            s = serve(Path(td), port=18772)
            threading.Thread(target=s.serve_forever, daemon=True).start()
            try:
                html = urlopen("http://127.0.0.1:18772/").read()
                self.assertIn(b"JuniorHome", html)
                req = Request("http://127.0.0.1:18772/", data=b"note=buy+oats", method="POST")
                msg = urlopen(req).read().decode()
                self.assertIn("ok=True", msg)
                self.assertIn("area=user", msg)
            finally:
                s.shutdown()
                s.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
