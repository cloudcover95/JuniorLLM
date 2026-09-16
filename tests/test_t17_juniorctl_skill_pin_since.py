"""T17 — juniorctl skill-pin since HDR is loopback-only and never fetches."""
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


class T17JuniorctlSkillPinSinceTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_since_empty_then_pin_then_load(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            missing = juniorctl.skill_pin_since(ZERO, str(td))
            self.assertFalse(missing["ok"], missing)
            self.assertEqual(missing["bind"], "127.0.0.1:8765")
            self.assertEqual(missing["op"], "since")
            self.assertEqual(missing["issues"], ["missing_hdr"])
            self.assertTrue(missing["empty"])
            self.assertFalse(missing["found"])
            self.assertEqual(missing["hdr"], ZERO)
            self.assertEqual(missing["from_hdr"], ZERO)
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
            tip0 = juniorctl.skill_pin_tip(str(td))
            self.assertTrue(tip0["ok"], tip0)
            zero = juniorctl.skill_pin_since(tip0["hdr"], str(td))
            self.assertTrue(zero["ok"], zero)
            self.assertFalse(zero["empty"])
            self.assertTrue(zero["found"])
            self.assertEqual(zero["last_op"], "pin")
            self.assertEqual(zero["rel"], rel)
            self.assertEqual(zero["name"], "plan-slice")
            self.assertEqual(zero["count"], 1)
            self.assertEqual(zero["total"], 1)
            self.assertEqual(zero["from_height"], 0)
            self.assertEqual(zero["entries"][0]["op"], "pin")
            self.assertEqual(zero["entries"][0]["height"], 0)
            self.assertEqual(zero["entries"][0]["sha256"], pinned["sha256"])
            self.assertEqual(zero["prev"], ZERO)
            self.assertEqual(len(zero["tip"] or ""), 64)
            self.assertEqual(zero["tip"], zero["entries"][0]["hdr"])
            self.assertEqual(zero["hdr"], tip0["hdr"])
            self.assertEqual(zero["from_hdr"], tip0["hdr"])
            self.assertNotEqual(zero["tip"], ZERO)
            self.assertTrue(zero["chain_ok"])
            self.assertNotIn("body", zero["entries"][0])
            loaded = juniorctl.skill_pin_load(rel, str(td))
            self.assertTrue(loaded["ok"], loaded)
            tip1 = juniorctl.skill_pin_tip(str(td))
            self.assertTrue(tip1["ok"], tip1)
            window = juniorctl.skill_pin_since(zero["hdr"], str(td))
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
            self.assertEqual(window["from_hdr"], zero["hdr"])
            only_tip = juniorctl.skill_pin_since(tip1["hdr"], str(td))
            self.assertTrue(only_tip["ok"], only_tip)
            self.assertEqual(only_tip["count"], 1)
            self.assertEqual(only_tip["entries"][0]["op"], "load")
            self.assertEqual(only_tip["entries"][0]["height"], 1)
            self.assertEqual(only_tip["from_hdr"], tip1["hdr"])
            ghost = "a" * 64
            absent = juniorctl.skill_pin_since(ghost, str(td))
            self.assertFalse(absent["ok"])
            self.assertEqual(absent["issues"], ["missing_hdr"])
            self.assertFalse(absent["found"])
            self.assertEqual(absent["from_hdr"], ghost)
            self.assertEqual(absent["hdr"], ghost)
            blob = json.dumps(window)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)
            self.assertNotIn("# Plan slice", blob)
            self.assertNotIn("body", blob)

    def test_cli_skill_pin_since_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            juniorctl.skill_pin_pin(rel, str(td))
            tip0 = juniorctl.skill_pin_tip(str(td))
            juniorctl.skill_pin_load(rel, str(td))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "since", tip0["hdr"], str(td)]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "since")
            self.assertEqual(payload["last_op"], "load")
            self.assertEqual(payload["count"], 2)
            self.assertEqual(payload["from_hdr"], tip0["hdr"])
            self.assertEqual(len(payload["tip"] or ""), 64)

    def test_cli_since_missing_and_bad_hdr(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "since", ZERO, str(td)]
                )
            self.assertEqual(code, 1)
            payload = json.loads(buf.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["issues"], ["missing_hdr"])
        empty = juniorctl.skill_pin_since("", "/tmp")
        self.assertFalse(empty["ok"])
        self.assertEqual(empty["issues"], ["empty_hdr"])
        bad = juniorctl.skill_pin_since("tip", "/tmp")
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["issues"], ["bad_hdr"])
        short = juniorctl.skill_pin_since("ab", "/tmp")
        self.assertFalse(short["ok"])
        self.assertEqual(short["issues"], ["bad_hdr"])
        heightish = juniorctl.skill_pin_since("0", "/tmp")
        self.assertFalse(heightish["ok"])
        self.assertEqual(heightish["issues"], ["bad_hdr"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = juniorctl.main(["juniorctl", "skill-pin", "since"])
        self.assertEqual(code, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["issues"], ["empty_hdr"])

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_since(ZERO, "/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_since(ZERO, "0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_since(ZERO, "../escape")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        pat = juniorctl.skill_pin_since(ZERO, "github_pat_secret")
        self.assertFalse(pat["ok"])

    def test_health_lists_skill_pin_since(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin since"', src)
        self.assertIn("skill_pin_since", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("docker.sock", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)
        health = juniorctl.health()
        self.assertIn("skill-pin since", health["cmds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
