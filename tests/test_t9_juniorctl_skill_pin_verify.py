"""T9 — juniorctl skill-pin verify is loopback-only and never fetches."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rails.linux import juniorctl

SAMPLE = """---
name: plan-slice
description: Start of every overnight run.
---
# Plan slice

Write 5 bullets before any git write.
"""


class T9JuniorctlSkillPinVerifyTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_verify_unpinned_then_pinned_then_mismatch(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            report = juniorctl.skill_pin_verify(str(td))
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["bind"], "127.0.0.1:8765")
            self.assertEqual(report["op"], "verify")
            self.assertFalse(report["fetch"])
            self.assertFalse(report["download"])
            self.assertFalse(report["docker_socket"])
            self.assertFalse(report["privileged"])
            self.assertFalse(report["exec"])
            self.assertTrue(report["chain_ok"])
            self.assertEqual(report["count"], 1)
            self.assertEqual(report["matched"], 0)
            self.assertEqual(report["skills"][0]["rel"], rel)
            self.assertFalse(report["skills"][0]["pinned"])
            self.assertFalse(report["skills"][0]["match"])
            self.assertEqual(report["skills"][0]["state"], "unpinned")
            pinned = juniorctl.skill_pin_pin(rel, str(td))
            self.assertTrue(pinned["ok"], pinned)
            report = juniorctl.skill_pin_verify(str(td))
            self.assertTrue(report["ok"], report)
            self.assertTrue(report["skills"][0]["pinned"])
            self.assertTrue(report["skills"][0]["match"])
            self.assertEqual(report["skills"][0]["state"], "ok")
            self.assertEqual(report["skills"][0]["sha256"], pinned["sha256"])
            self.assertEqual(report["matched"], 1)
            self.assertTrue(report["chain_ok"])
            (td / rel).write_text(SAMPLE + "\nmutated\n", encoding="utf-8")
            report = juniorctl.skill_pin_verify(str(td))
            self.assertFalse(report["ok"])
            self.assertEqual(report["issues"], ["pin_mismatch"])
            self.assertEqual(report["skills"][0]["state"], "pin_mismatch")
            self.assertFalse(report["skills"][0]["match"])
            blob = json.dumps(report)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)

    def test_cli_skill_pin_verify_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            juniorctl.skill_pin_pin(rel, str(td))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(["juniorctl", "skill-pin", "verify", str(td)])
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "verify")
            self.assertEqual(payload["matched"], 1)

    def test_cli_verify_mismatch_exits_one(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            juniorctl.skill_pin_pin(rel, str(td))
            (td / rel).write_text(SAMPLE + "\nmutated\n", encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(["juniorctl", "skill-pin", "verify", str(td)])
            self.assertEqual(code, 1)
            payload = json.loads(buf.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["issues"], ["pin_mismatch"])

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_verify("/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_verify("0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_verify("../escape")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        pat = juniorctl.skill_pin_verify("github_pat_secret")
        self.assertFalse(pat["ok"])

    def test_health_lists_skill_pin_verify(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin verify"', src)
        self.assertIn("skill_pin_verify", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("docker.sock", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
