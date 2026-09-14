"""T12 — juniorctl skill-pin log is loopback-only and never fetches."""
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


class T12JuniorctlSkillPinLogTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_log_empty_then_pin_then_load(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            report = juniorctl.skill_pin_log(str(td))
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["bind"], "127.0.0.1:8765")
            self.assertEqual(report["op"], "log")
            self.assertEqual(report["tip"], ZERO)
            self.assertEqual(report["hdr"], ZERO)
            self.assertTrue(report["empty"])
            self.assertEqual(report["count"], 0)
            self.assertEqual(report["entries"], [])
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
            report = juniorctl.skill_pin_log(str(td))
            self.assertTrue(report["ok"], report)
            self.assertFalse(report["empty"])
            self.assertEqual(report["last_op"], "pin")
            self.assertEqual(report["rel"], rel)
            self.assertEqual(report["name"], "plan-slice")
            self.assertEqual(report["count"], 1)
            self.assertEqual(len(report["entries"]), 1)
            self.assertEqual(report["entries"][0]["op"], "pin")
            self.assertEqual(report["entries"][0]["height"], 0)
            self.assertEqual(report["entries"][0]["sha256"], pinned["sha256"])
            self.assertEqual(report["tip"], report["entries"][0]["hdr"])
            self.assertNotEqual(report["tip"], ZERO)
            self.assertTrue(report["chain_ok"])
            self.assertNotIn("body", report["entries"][0])
            loaded = juniorctl.skill_pin_load(rel, str(td))
            self.assertTrue(loaded["ok"], loaded)
            after = juniorctl.skill_pin_log(str(td))
            self.assertTrue(after["ok"], after)
            self.assertEqual(after["last_op"], "load")
            self.assertEqual(after["count"], 2)
            self.assertEqual(len(after["entries"]), 2)
            self.assertEqual(after["entries"][0]["op"], "pin")
            self.assertEqual(after["entries"][1]["op"], "load")
            self.assertEqual(after["entries"][1]["prev"], after["entries"][0]["hdr"])
            self.assertEqual(after["tip"], after["entries"][1]["hdr"])
            self.assertNotEqual(after["tip"], report["tip"])
            blob = json.dumps(after)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)
            self.assertNotIn("# Plan slice", blob)
            self.assertNotIn("body", blob)

    def test_cli_skill_pin_log_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            juniorctl.skill_pin_pin(rel, str(td))
            juniorctl.skill_pin_load(rel, str(td))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(["juniorctl", "skill-pin", "log", str(td)])
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "log")
            self.assertEqual(payload["last_op"], "load")
            self.assertEqual(payload["count"], 2)
            self.assertEqual(len(payload["tip"] or ""), 64)

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_log("/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_log("0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_log("../escape")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        pat = juniorctl.skill_pin_log("github_pat_secret")
        self.assertFalse(pat["ok"])

    def test_health_lists_skill_pin_log(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin log"', src)
        self.assertIn("skill_pin_log", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("docker.sock", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)
        health = juniorctl.health()
        self.assertIn("skill-pin log", health["cmds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
