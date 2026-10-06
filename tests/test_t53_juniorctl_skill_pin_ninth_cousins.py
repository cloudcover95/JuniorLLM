"""T53 — juniorctl skill-pin ninth-cousins is loopback-only and never fetches."""
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
G8 = "great_great_great_great_great_great_great_great_grandparent_hdr"
G8H = "great_great_great_great_great_great_great_great_grandparent_height"


class T53JuniorctlSkillPinNinthCousinsTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def _eleven(self, td: Path, rel: str) -> None:
        for _ in range(5):
            juniorctl.skill_pin_pin(rel, str(td))
            juniorctl.skill_pin_load(rel, str(td))
        juniorctl.skill_pin_pin(rel, str(td))

    def test_9c_empty_then_eleven_high(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            missing = juniorctl.skill_pin_ninth_cousins(ZERO, str(td))
            self.assertFalse(missing["ok"], missing)
            self.assertEqual(missing["bind"], "127.0.0.1:8765")
            self.assertEqual(missing["op"], "ninth-cousins")
            self.assertEqual(missing["issues"], ["missing_hdr"])
            self.assertTrue(missing["empty"])
            self.assertFalse(missing["found"])
            self.assertFalse(missing["is_only"])
            self.assertFalse(missing["fetch"])
            self.assertFalse(missing["exec"])
            self.assertFalse(missing["docker_socket"])
            self.assertNotIn("body", missing)
            self._eleven(td, rel)
            tip = juniorctl.skill_pin_tip(str(td))
            gen = juniorctl.skill_pin_genesis(str(td))
            of_h10 = juniorctl.skill_pin_ninth_cousins(tip["hdr"], str(td))
            self.assertTrue(of_h10["ok"], of_h10)
            self.assertFalse(of_h10["found"])
            self.assertTrue(of_h10["is_only"])
            self.assertEqual(of_h10["issues"], [])
            self.assertEqual(of_h10["total"], 11)
            self.assertEqual(of_h10["count"], 0)
            self.assertEqual(of_h10["child_hdr"], tip["hdr"])
            self.assertEqual(of_h10["child_height"], 10)
            self.assertEqual(of_h10[G8], gen["hdr"])
            self.assertEqual(of_h10[G8H], 0)
            blob = json.dumps(of_h10)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)
            self.assertNotIn("body", blob)

    def test_cli_skill_pin_ninth_cousins_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            self._eleven(td, rel)
            gen = juniorctl.skill_pin_genesis(str(td))
            tip = juniorctl.skill_pin_tip(str(td))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "ninth-cousins", tip["hdr"], str(td)]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "ninth-cousins")
            self.assertTrue(payload["is_only"])
            self.assertEqual(payload["total"], 11)
            self.assertEqual(payload["child_height"], 10)
            self.assertEqual(payload[G8], gen["hdr"])
            self.assertEqual(payload[G8H], 0)
            self.assertFalse(payload["fetch"])
            self.assertFalse(payload["exec"])
            self.assertNotIn("body", payload)

    def test_cli_9c_missing_and_bad_hdr(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "ninth-cousins", ZERO, str(td)]
                )
            self.assertEqual(code, 1)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["issues"], ["missing_hdr"])
        empty = juniorctl.skill_pin_ninth_cousins("", "/tmp")
        self.assertEqual(empty["issues"], ["empty_hdr"])
        bad = juniorctl.skill_pin_ninth_cousins("tip", "/tmp")
        self.assertEqual(bad["issues"], ["bad_hdr"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = juniorctl.main(["juniorctl", "skill-pin", "ninth-cousins"])
        self.assertEqual(code, 1)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["issues"], ["empty_hdr"])

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_ninth_cousins(ZERO, "/var/run/docker.sock")
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_ninth_cousins(ZERO, "0.0.0.0")
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_ninth_cousins(ZERO, "../escape")
        self.assertEqual(esc["issues"], ["path_escape"])

    def test_health_lists_skill_pin_ninth_cousins(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin ninth-cousins"', src)
        self.assertIn("skill_pin_ninth_cousins", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        impl = (ROOT / "rails" / "linux" / "ctl_skillpin_ninth_cousins.py").read_text(encoding="utf-8")
        self.assertNotIn("eval(", impl)
        self.assertNotIn("exec(", impl)
        extra = (ROOT / "rails" / "linux" / "ctl_cli_t53.py").read_text(encoding="utf-8")
        self.assertIn("skill_pin_ninth_cousins", extra)
        self.assertNotIn("eval(", extra)
        self.assertNotIn("exec(", extra)
        health = juniorctl.health()
        self.assertIn("skill-pin ninth-cousins", health["cmds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
