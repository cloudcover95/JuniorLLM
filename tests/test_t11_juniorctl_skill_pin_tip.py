"""T11 — juniorctl skill-pin tip is loopback-only and never fetches."""
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

ZERO = "0" * 64


class T11JuniorctlSkillPinTipTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_tip_empty_then_pin_then_load(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            report = juniorctl.skill_pin_tip(str(td))
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["bind"], "127.0.0.1:8765")
            self.assertEqual(report["op"], "tip")
            self.assertEqual(report["tip"], ZERO)
            self.assertEqual(report["hdr"], ZERO)
            self.assertTrue(report["empty"])
            self.assertEqual(report["count"], 0)
            self.assertEqual(report["last_op"], "")
            self.assertFalse(report["fetch"])
            self.assertFalse(report["download"])
            self.assertFalse(report["docker_socket"])
            self.assertFalse(report["privileged"])
            self.assertFalse(report["exec"])
            self.assertTrue(report["chain_ok"])
            self.assertNotIn("body", report)
            pinned = juniorctl.skill_pin_pin(rel, str(td))
            self.assertTrue(pinned["ok"], pinned)
            report = juniorctl.skill_pin_tip(str(td))
            self.assertTrue(report["ok"], report)
            self.assertFalse(report["empty"])
            self.assertEqual(report["last_op"], "pin")
            self.assertEqual(report["rel"], rel)
            self.assertEqual(report["name"], "plan-slice")
            self.assertEqual(report["sha256"], pinned["sha256"])
            self.assertEqual(len(report["tip"] or ""), 64)
            self.assertEqual(report["tip"], report["hdr"])
            self.assertNotEqual(report["tip"], ZERO)
            self.assertEqual(report["count"], 1)
            self.assertTrue(report["chain_ok"])
            loaded = juniorctl.skill_pin_load(rel, str(td))
            self.assertTrue(loaded["ok"], loaded)
            after = juniorctl.skill_pin_tip(str(td))
            self.assertTrue(after["ok"], after)
            self.assertEqual(after["last_op"], "load")
            self.assertEqual(after["count"], 2)
            self.assertNotEqual(after["tip"], report["tip"])
            blob = json.dumps(after)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)

    def test_cli_skill_pin_tip_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            juniorctl.skill_pin_pin(rel, str(td))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(["juniorctl", "skill-pin", "tip", str(td)])
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "tip")
            self.assertEqual(payload["last_op"], "pin")
            self.assertEqual(len(payload["tip"] or ""), 64)

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_tip("/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_tip("0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_tip("../escape")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        pat = juniorctl.skill_pin_tip("github_pat_secret")
        self.assertFalse(pat["ok"])

    def test_health_lists_skill_pin_tip(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin tip"', src)
        self.assertIn("skill_pin_tip", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("docker.sock", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)
        health = juniorctl.health()
        self.assertIn("skill-pin tip", health["cmds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
