"""T8 — juniorctl skill-pin pin is loopback-only and never fetches."""
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


class T8JuniorctlSkillPinPinTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_pin_then_list_and_load(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            listed = juniorctl.skill_pin_list(str(td))
            self.assertTrue(listed["ok"], listed)
            self.assertFalse(listed["skills"][0]["pinned"])
            report = juniorctl.skill_pin_pin(rel, str(td))
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["op"], "pin")
            self.assertEqual(report["name"], "plan-slice")
            self.assertEqual(report["rel"], rel)
            self.assertEqual(report["bind"], "127.0.0.1:8765")
            self.assertEqual(len(report["sha256"] or ""), 64)
            self.assertTrue(report["chain_ok"])
            self.assertFalse(report["fetch"])
            self.assertFalse(report["download"])
            self.assertFalse(report["docker_socket"])
            self.assertFalse(report["privileged"])
            self.assertFalse(report["exec"])
            listed = juniorctl.skill_pin_list(str(td))
            self.assertTrue(listed["skills"][0]["pinned"])
            self.assertEqual(listed["skills"][0]["sha256"], report["sha256"])
            loaded = juniorctl.skill_pin_load(rel, str(td))
            self.assertTrue(loaded["ok"], loaded)
            self.assertEqual(loaded["body"], SAMPLE)
            self.assertEqual(loaded["sha256"], report["sha256"])
            blob = json.dumps(report)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)

    def test_pin_banned_body_and_missing_file(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            bad = td / "skills" / "evil" / "SKILL.md"
            bad.parent.mkdir(parents=True)
            bad.write_text("---\nname: evil\n---\neval(user)\n", encoding="utf-8")
            report = juniorctl.skill_pin_pin("skills/evil/SKILL.md", str(td))
            self.assertFalse(report["ok"])
            self.assertEqual(report["issues"], ["banned_body"])
            missing = juniorctl.skill_pin_pin("skills/gone/SKILL.md", str(td))
            self.assertFalse(missing["ok"])
            self.assertEqual(missing["issues"], ["not_skill_md"])

    def test_cli_skill_pin_pin_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(["juniorctl", "skill-pin", "pin", rel, str(td)])
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "pin")
            self.assertEqual(payload["name"], "plan-slice")

    def test_cli_pin_missing_rel_is_denied(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = juniorctl.main(["juniorctl", "skill-pin", "pin"])
        self.assertEqual(code, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["issues"], ["empty_path"])

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_pin("skills/x/SKILL.md", "/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_pin("skills/x/SKILL.md", "0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_pin("../escape/SKILL.md", "/tmp")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        pat = juniorctl.skill_pin_pin("skills/x/SKILL.md", "github_pat_secret")
        self.assertFalse(pat["ok"])

    def test_health_lists_skill_pin_pin(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin pin"', src)
        self.assertIn("skill_pin_pin", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("docker.sock", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
