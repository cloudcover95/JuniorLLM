"""T16 — juniorctl skill-pin range FROM TO is loopback-only and never fetches."""
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


class T16JuniorctlSkillPinRangeTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_range_empty_then_pin_then_load(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            missing = juniorctl.skill_pin_range("0", "0", str(td))
            self.assertFalse(missing["ok"], missing)
            self.assertEqual(missing["bind"], "127.0.0.1:8765")
            self.assertEqual(missing["op"], "range")
            self.assertEqual(missing["issues"], ["missing_range"])
            self.assertTrue(missing["empty"])
            self.assertFalse(missing["found"])
            self.assertEqual(missing["from"], 0)
            self.assertEqual(missing["to"], 0)
            self.assertEqual(missing["count"], 0)
            self.assertEqual(missing["total"], 0)
            self.assertEqual(missing["entries"], [])
            self.assertEqual(missing["tip"], ZERO)
            self.assertFalse(missing["fetch"])
            self.assertFalse(missing["download"])
            self.assertFalse(missing["docker_socket"])
            self.assertFalse(missing["privileged"])
            self.assertFalse(missing["exec"])
            self.assertNotIn("body", missing)
            pinned = juniorctl.skill_pin_pin(rel, str(td))
            self.assertTrue(pinned["ok"], pinned)
            zero = juniorctl.skill_pin_range("0", "0", str(td))
            self.assertTrue(zero["ok"], zero)
            self.assertFalse(zero["empty"])
            self.assertTrue(zero["found"])
            self.assertEqual(zero["last_op"], "pin")
            self.assertEqual(zero["rel"], rel)
            self.assertEqual(zero["name"], "plan-slice")
            self.assertEqual(zero["count"], 1)
            self.assertEqual(zero["total"], 1)
            self.assertEqual(zero["from"], 0)
            self.assertEqual(zero["to"], 0)
            self.assertEqual(zero["entries"][0]["op"], "pin")
            self.assertEqual(zero["entries"][0]["height"], 0)
            self.assertEqual(zero["entries"][0]["sha256"], pinned["sha256"])
            self.assertEqual(zero["prev"], ZERO)
            self.assertEqual(len(zero["tip"] or ""), 64)
            self.assertEqual(zero["tip"], zero["entries"][0]["hdr"])
            self.assertNotEqual(zero["tip"], ZERO)
            self.assertTrue(zero["chain_ok"])
            self.assertNotIn("body", zero["entries"][0])
            loaded = juniorctl.skill_pin_load(rel, str(td))
            self.assertTrue(loaded["ok"], loaded)
            window = juniorctl.skill_pin_range("0", "1", str(td))
            self.assertTrue(window["ok"], window)
            self.assertEqual(window["last_op"], "load")
            self.assertEqual(window["count"], 2)
            self.assertEqual(window["total"], 2)
            self.assertEqual(len(window["entries"]), 2)
            self.assertEqual(window["entries"][0]["op"], "pin")
            self.assertEqual(window["entries"][0]["height"], 0)
            self.assertEqual(window["entries"][1]["op"], "load")
            self.assertEqual(window["entries"][1]["height"], 1)
            self.assertEqual(window["entries"][1]["prev"], window["entries"][0]["hdr"])
            self.assertEqual(window["height"], loaded["height"])
            self.assertEqual(window["tip"], window["entries"][1]["hdr"])
            only_one = juniorctl.skill_pin_range("1", "1", str(td))
            self.assertTrue(only_one["ok"], only_one)
            self.assertEqual(only_one["count"], 1)
            self.assertEqual(only_one["entries"][0]["op"], "load")
            self.assertEqual(only_one["entries"][0]["height"], 1)
            ghost = juniorctl.skill_pin_range("9", "9", str(td))
            self.assertFalse(ghost["ok"])
            self.assertEqual(ghost["issues"], ["missing_range"])
            self.assertFalse(ghost["found"])
            self.assertEqual(ghost["from"], 9)
            self.assertEqual(ghost["to"], 9)
            blob = json.dumps(window)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)
            self.assertNotIn("# Plan slice", blob)
            self.assertNotIn("body", blob)

    def test_cli_skill_pin_range_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            juniorctl.skill_pin_pin(rel, str(td))
            juniorctl.skill_pin_load(rel, str(td))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "range", "0", "1", str(td)]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "range")
            self.assertEqual(payload["last_op"], "load")
            self.assertEqual(payload["count"], 2)
            self.assertEqual(payload["from"], 0)
            self.assertEqual(payload["to"], 1)
            self.assertEqual(len(payload["tip"] or ""), 64)

    def test_cli_range_missing_and_bad_bounds(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "range", "0", "0", str(td)]
                )
            self.assertEqual(code, 1)
            payload = json.loads(buf.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["issues"], ["missing_range"])
        empty_from = juniorctl.skill_pin_range("", "1", "/tmp")
        self.assertFalse(empty_from["ok"])
        self.assertEqual(empty_from["issues"], ["empty_from"])
        empty_to = juniorctl.skill_pin_range("0", "", "/tmp")
        self.assertFalse(empty_to["ok"])
        self.assertEqual(empty_to["issues"], ["empty_to"])
        bad_from = juniorctl.skill_pin_range("tip", "1", "/tmp")
        self.assertFalse(bad_from["ok"])
        self.assertEqual(bad_from["issues"], ["bad_from"])
        bad_to = juniorctl.skill_pin_range("0", "tip", "/tmp")
        self.assertFalse(bad_to["ok"])
        self.assertEqual(bad_to["issues"], ["bad_to"])
        neg = juniorctl.skill_pin_range("-1", "1", "/tmp")
        self.assertFalse(neg["ok"])
        self.assertEqual(neg["issues"], ["bad_from"])
        inverted = juniorctl.skill_pin_range("2", "1", "/tmp")
        self.assertFalse(inverted["ok"])
        self.assertEqual(inverted["issues"], ["inverted_range"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = juniorctl.main(["juniorctl", "skill-pin", "range"])
        self.assertEqual(code, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["issues"], ["empty_from"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = juniorctl.main(["juniorctl", "skill-pin", "range", "0"])
        self.assertEqual(code, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["issues"], ["empty_to"])

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_range("0", "1", "/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_range("0", "1", "0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_range("0", "1", "../escape")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        pat = juniorctl.skill_pin_range("0", "1", "github_pat_secret")
        self.assertFalse(pat["ok"])

    def test_health_lists_skill_pin_range(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin range"', src)
        self.assertIn("skill_pin_range", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("docker.sock", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)
        health = juniorctl.health()
        self.assertIn("skill-pin range", health["cmds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
