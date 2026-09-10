"""C5 — juniorctl oci validate is loopback-only and never fetches weights."""
from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rails.linux import juniorctl


class C5JuniorctlOciTests(unittest.TestCase):
    def test_oci_validate_report(self):
        report = juniorctl.oci_validate()
        self.assertTrue(report["ok"], report["issues"])
        self.assertEqual(report["bind"], "127.0.0.1:8765")
        self.assertTrue(report["rootless"])
        self.assertFalse(report["privileged"])
        self.assertFalse(report["docker_socket"])
        self.assertEqual(report["unit"], "bitnetd-rootless")
        self.assertEqual(report["unit_status"], "ready")
        self.assertFalse(report["gemma"]["fetch"])
        self.assertEqual(report["gemma"]["port"], "JuniorGemma4-4B")
        raw = json.dumps(report)
        self.assertNotIn("0.0.0.0", raw)
        self.assertNotIn("docker.sock", raw)

    def test_cli_oci_validate_exit_zero(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = juniorctl.main(["juniorctl", "oci", "validate"])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["bind"], "127.0.0.1:8765")

    def test_cli_oci_unknown_sub_is_usage(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = juniorctl.main(["juniorctl", "oci", "run"])
        self.assertEqual(code, 2)
        self.assertEqual(buf.getvalue(), "")

    def test_health_lists_oci_validate(self):
        h = juniorctl.health()
        self.assertIn("oci validate", h["cmds"])
        self.assertEqual(h["bitnetd"], "127.0.0.1:8765")

    def test_source_has_no_eval_exec_or_wildcard(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertIn("oci_validate", src)
        self.assertNotIn("docker.sock", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
