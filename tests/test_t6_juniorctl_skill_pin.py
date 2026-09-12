"""T6 — juniorctl skill-pin list is loopback-only and never fetches."""
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

from junior_aie.skill_pin import SkillPins
from rails.linux import juniorctl

SAMPLE = """---
name: plan-slice
description: Start of every overnight run.
---
# Plan slice

Write 5 bullets before any git write.
"""


class T6JuniorctlSkillPinTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_list_unpinned_then_pinned(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            report = juniorctl.skill_pin_list(str(td))
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["bind"], "127.0.0.1:8765")
            self.assertFalse(report["fetch"])
            self.assertFalse(report["download"])
            self.assertFalse(report["docker_socket"])
            self.assertFalse(report["privileged"])
            self.assertEqual(report["count"], 1)
            self.assertEqual(report["skills"][0]["rel"], rel)
            self.assertFalse(report["skills"][0]["pinned"])
            pins = SkillPins(td)
            pins.pin(rel)
            report = juniorctl.skill_pin_list(str(td))
            self.assertTrue(report["ok"])
            self.assertTrue(report["skills"][0]["pinned"])
            self.assertEqual(report["skills"][0]["name"], "plan-slice")
            self.assertEqual(len(report["skills"][0]["sha256"] or ""), 64)
            self.assertTrue(report["chain_ok"])
            blob = json.dumps(report)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)

    def test_cli_skill_pin_list_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            self._tree(td)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(["juniorctl", "skill-pin", "list", str(td)])
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["count"], 1)

    def test_cli_unknown_sub_is_usage(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = juniorctl.main(["juniorctl", "skill-pin", "fetch"])
        self.assertEqual(code, 2)
        self.assertEqual(buf.getvalue(), "")

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_list("/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_list("0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_list("../escape")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        pat = juniorctl.skill_pin_list("github_pat_secret")
        self.assertFalse(pat["ok"])

    def test_health_lists_skill_pin_list(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin list"', src)
        self.assertIn("skill_pin_list", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("docker.sock", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
