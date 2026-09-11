"""T4 — JuniorFileLedger local create/read."""
from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptations.gemma4.ondisk_bind import notes as gemma_notes
from junior_aie.files_ledger import CAP_BYTES, FilesLedger, LedgerDenied, too_big
from ports.ondisk import probe


class T4FilesLedgerTests(unittest.TestCase):
    def test_create_then_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            led = FilesLedger(Path(td))
            row = led.create("notes/hello.txt", b"flagstaff dry")
            self.assertEqual(row.op, "create")
            self.assertEqual(row.size, 13)
            data, rrow = led.read("notes/hello.txt")
            self.assertEqual(data, b"flagstaff dry")
            self.assertEqual(rrow.op, "read")
            self.assertEqual(rrow.sha256, row.sha256)
            self.assertTrue(led.verify_chain())

    def test_refuse_path_escape_and_docker_sock(self):
        with tempfile.TemporaryDirectory() as td:
            led = FilesLedger(Path(td))
            with self.assertRaises(LedgerDenied):
                led.create("../escape.txt", b"x")
            with self.assertRaises(LedgerDenied):
                led.create("/etc/passwd", b"x")
            with self.assertRaises(LedgerDenied):
                led.create("var/run/docker.sock", b"x")
            with self.assertRaises(LedgerDenied):
                led.read("missing.txt")

    def test_cap_helper_never_allocates_8gb(self):
        self.assertTrue(too_big(CAP_BYTES + 1))
        self.assertFalse(too_big(16))
        self.assertEqual(CAP_BYTES, 8 * 1024 * 1024 * 1024)

    def test_b2_ondisk_probe_never_downloads(self):
        ports = probe(ROOT / "missing-models")
        self.assertTrue(ports)
        self.assertTrue(all(not p.present for p in ports))
        src = (ROOT / "ports" / "ondisk.py").read_text(encoding="utf-8")
        self.assertIn("Never fetch", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)

    def test_b3_gemma_ondisk_notes_only(self):
        info = gemma_notes(ROOT / "missing-models")
        self.assertFalse(info["present"])
        self.assertFalse(info["fetch"])
        self.assertEqual(info["backend"], "JuniorBitNetFieldCore")
        src = (ROOT / "adaptations" / "gemma4" / "ondisk_bind.py").read_text(encoding="utf-8")
        self.assertIn("Never fetch", src)
        self.assertNotIn("urllib", src)

    def test_module_stdlib_and_no_eval(self):
        src = (ROOT / "junior_aie" / "files_ledger.py").read_text(encoding="utf-8")
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("docker.sock", src.split("DENY", 1)[0])
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)
        tree = ast.parse(src)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module.split(".")[0])
        self.assertTrue(set(imported) <= {"__future__", "hashlib", "json", "dataclasses", "datetime", "pathlib"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
