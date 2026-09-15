"""T14 — juniorctl skill-pin get HEIGHT is loopback-only and never fetches."""
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


class T14JuniorctlSkillPinGetTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_get_empty_then_pin_then_load(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            missing = juniorctl.skill_pin_get("0", str(td))
            self.assertFalse(missing["ok"], missing)
            self.assertEqual(missing["bind"], "127.0.0.1:8765")
            self.assertEqual(missing["op"], "get")
            self.assertEqual(missing["issues"], ["missing_height"])
            self.assertTrue(missing["empty"])
            self.assertFalse(missing["found"])
            self.assertEqual(missing["height"], 0)
            self.assertEqual(missing["count"], 0)
            self.assertEqual(missing["tip"], ZERO)
            self.assertEqual(missing["hdr"], ZERO)
            self.assertFalse(missing["fetch"])
            self.assertFalse(missing["download"])
            self.assertFalse(missing["docker_socket"])
            self.assertFalse(missing["privileged"])
            self.assertFalse(missing["exec"])
            self.assertNotIn("body", missing)
            self.assertNotIn("entries", missing)
            pinned = juniorctl.skill_pin_pin(rel, str(td))
            self.assertTrue(pinned["ok"], pinned)
            zero = juniorctl.skill_pin_get("0", str(td))
            self.assertTrue(zero["ok"], zero)
            self.assertFalse(zero["empty"])
            self.assertTrue(zero["found"])
            self.assertEqual(zero["last_op"], "pin")
            self.assertEqual(zero["rel"], rel)
            self.assertEqual(zero["name"], "plan-slice")
            self.assertEqual(zero["count"], 1)
            self.assertEqual(zero["height"], 0)
            self.assertEqual(zero["sha256"], pinned["sha256"])
            self.assertEqual(zero["prev"], ZERO)
            self.assertEqual(len(zero["tip"] or ""), 64)
            self.assertEqual(zero["tip"], zero["hdr"])
            self.assertNotEqual(zero["tip"], ZERO)
            self.assertTrue(zero["chain_ok"])
            self.assertNotIn("entries", zero)
            loaded = juniorctl.skill_pin_load(rel, str(td))
            self.assertTrue(loaded["ok"], loaded)
            one = juniorctl.skill_pin_get("1", str(td))
            self.assertTrue(one["ok"], one)
            self.assertEqual(one["last_op"], "load")
            self.assertEqual(one["count"], 2)
            self.assertEqual(one["height"], 1)
            self.assertEqual(one["height"], loaded["height"])
            self.assertEqual(one["prev"], zero["hdr"])
            self.assertEqual(one["tip"], one["hdr"])
            self.assertNotEqual(one["tip"], zero["tip"])
            still_zero = juniorctl.skill_pin_get("0", str(td))
            self.assertTrue(still_zero["ok"], still_zero)
            self.assertEqual(still_zero["last_op"], "pin")
            self.assertEqual(still_zero["height"], 0)
            self.assertEqual(still_zero["hdr"], zero["hdr"])
            self.assertEqual(still_zero["tip"], one["tip"])
            absent = juniorctl.skill_pin_get("9", str(td))
            self.assertFalse(absent["ok"])
            self.assertEqual(absent["issues"], ["missing_height"])
            self.assertFalse(absent["found"])
            self.assertEqual(absent["height"], 9)
            blob = json.dumps(one)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)
            self.assertNotIn("# Plan slice", blob)
            self.assertNotIn("body", blob)
            self.assertNotIn("entries", blob)

    def test_cli_skill_pin_get_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            juniorctl.skill_pin_pin(rel, str(td))
            juniorctl.skill_pin_load(rel, str(td))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(["juniorctl", "skill-pin", "get", "1", str(td)])
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "get")
            self.assertEqual(payload["last_op"], "load")
            self.assertEqual(payload["count"], 2)
            self.assertEqual(payload["height"], 1)
            self.assertEqual(len(payload["tip"] or ""), 64)

    def test_cli_get_missing_and_bad_height(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(["juniorctl", "skill-pin", "get", "0", str(td)])
            self.assertEqual(code, 1)
            payload = json.loads(buf.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["issues"], ["missing_height"])
        empty = juniorctl.skill_pin_get("", "/tmp")
        self.assertFalse(empty["ok"])
        self.assertEqual(empty["issues"], ["empty_height"])
        bad = juniorctl.skill_pin_get("tip", "/tmp")
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["issues"], ["bad_height"])
        neg = juniorctl.skill_pin_get("-1", "/tmp")
        self.assertFalse(neg["ok"])
        self.assertEqual(neg["issues"], ["bad_height"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = juniorctl.main(["juniorctl", "skill-pin", "get"])
        self.assertEqual(code, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["issues"], ["empty_height"])

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_get("0", "/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_get("0", "0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_get("0", "../escape")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        pat = juniorctl.skill_pin_get("0", "github_pat_secret")
        self.assertFalse(pat["ok"])

    def test_health_lists_skill_pin_get(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin get"', src)
        self.assertIn("skill_pin_get", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("docker.sock", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)
        health = juniorctl.health()
        self.assertIn("skill-pin get", health["cmds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
