from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.second_brain import note
from rails.linux.hook import serve


class HookTests(unittest.TestCase):
    def test_brain(self):
        with tempfile.TemporaryDirectory() as td:
            p = note(Path(td), "hello")
            self.assertTrue(p.is_file())
            self.assertIn("JuniorHome", p.read_text())

    def test_http(self):
        srv = serve("127.0.0.1", 18770)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            with urlopen("http://127.0.0.1:18770/llm", timeout=5) as r:
                card = json.loads(r.read())
            self.assertEqual(card["port"], "JuniorBitNetFieldCore")
            req = Request(
                "http://127.0.0.1:18770/inject",
                data=json.dumps({"text": "flagstaff", "consent": True}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=8) as r:
                row = json.loads(r.read())
            self.assertTrue(row["ok"])
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
