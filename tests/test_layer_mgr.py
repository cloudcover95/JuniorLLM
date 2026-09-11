from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junior_bitnet.coolstore import build
from ports.layer_mgr import pick_eos, report
from ports.registry import pick


class LayerMgrTests(unittest.TestCase):
    def test_keywords_win(self):
        self.assertEqual(pick_eos("dxf title", 16, "DENSE").name, "JuniorBitNetDraft")
        self.assertEqual(pick_eos("fable safety", 16, "SPARSE").name, "JuniorFable")

    def test_phase_routes(self):
        tab = build()
        self.assertEqual(pick_eos("", 16, "SPARSE", tab).name, "JuniorBitNetFieldCore")
        self.assertEqual(pick_eos("", 16, "DENSE", tab).name, "JuniorAstraReason")

    def test_old_pick_untouched(self):
        self.assertEqual(pick("cad drawing", 8).name, "JuniorBitNetDraft")

    def test_report(self):
        rows = report(build(), 16)
        self.assertEqual(len(rows), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
