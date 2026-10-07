"""T55 — juniorctl skill-pin tenth-cousins is loopback-only and never fetches."""
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
G9 = "great_great_great_great_great_great_great_great_great_grandparent_hdr"
G9H = "great_great_great_great_great_great_great_great_great_grandparent_height"


class T55JuniorctlSkillPinTenthCousinsTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def _twelve(self, td: Path, rel: str) -> None:
        for _ in range(6):
            juniorctl.skill_pin_pin(rel, str(td))
            juniorctl.skill_pin_load(rel, str(td))

    def test_10c_empty_then_twelve_high(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            missing = juniorctl.skill_pin_tenth_cousins(ZERO, str(td))
            self.assertFalse(missing["ok"], missing)
            self.assertEqual(missing["bind"], "127.0.0.1:8765")
            self.assertEqual(missing["op"], "tenth-cousins")
            self.assertEqual(missing["issues"], ["missing_hdr"])
            self.assertTrue(missing["empty"])
            self.assertFalse(missing["found"])
            self.assertFalse(missing["is_only"])
            self.assertFalse(missing["is_genesis"])
            self.assertEqual(missing["hdr"], ZERO)
            self.assertEqual(missing["prev"], ZERO)
            self.assertEqual(missing["sha256"], ZERO)
            self.assertEqual(missing["genesis"], ZERO)
            self.assertEqual(missing["height"], -1)
            self.assertEqual(missing["count"], 0)
            self.assertEqual(missing["total"], 0)
            self.assertEqual(missing["tip"], ZERO)
            self.assertEqual(missing["child_hdr"], ZERO)
            self.assertEqual(missing["child_height"], -1)
            self.assertEqual(missing["parent_hdr"], ZERO)
            self.assertEqual(missing["grandparent_hdr"], ZERO)
            self.assertEqual(missing["great_grandparent_hdr"], ZERO)
            self.assertEqual(missing["great_great_grandparent_hdr"], ZERO)
            self.assertEqual(missing["great_great_great_grandparent_hdr"], ZERO)
            self.assertEqual(missing["great_great_great_great_grandparent_hdr"], ZERO)
            self.assertEqual(missing["great_great_great_great_great_grandparent_hdr"], ZERO)
            self.assertEqual(missing["great_great_great_great_great_great_grandparent_hdr"], ZERO)
            self.assertEqual(missing["great_great_great_great_great_great_great_grandparent_hdr"], ZERO)
            self.assertEqual(
                missing["great_great_great_great_great_great_great_great_grandparent_hdr"], ZERO
            )
            self.assertEqual(missing[G9], ZERO)
            self.assertEqual(missing[G9H], -1)
            self.assertEqual(missing["entries"], [])
            self.assertFalse(missing["fetch"])
            self.assertFalse(missing["download"])
            self.assertFalse(missing["docker_socket"])
            self.assertFalse(missing["privileged"])
            self.assertFalse(missing["exec"])
            self.assertNotIn("body", missing)

            self._twelve(td, rel)
            tip = juniorctl.skill_pin_tip(str(td))
            gen = juniorctl.skill_pin_genesis(str(td))
            of_h11 = juniorctl.skill_pin_tenth_cousins(tip["hdr"], str(td))
            self.assertTrue(of_h11["ok"], of_h11)
            self.assertFalse(of_h11["found"])
            self.assertTrue(of_h11["is_only"])
            self.assertFalse(of_h11["is_genesis"])
            self.assertEqual(of_h11["issues"], [])
            self.assertEqual(of_h11["total"], 12)
            self.assertEqual(of_h11["count"], 0)
            self.assertEqual(of_h11["entries"], [])
            self.assertEqual(of_h11["child_hdr"], tip["hdr"])
            self.assertEqual(of_h11["child_height"], 11)
            self.assertEqual(of_h11[G9], gen["hdr"])
            self.assertEqual(of_h11[G9H], 0)
            self.assertEqual(of_h11["genesis"], gen["hdr"])
            blob = json.dumps(of_h11)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)
            self.assertNotIn("body", blob)

    def test_cli_skill_pin_tenth_cousins_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            self._twelve(td, rel)
            gen = juniorctl.skill_pin_genesis(str(td))
            tip = juniorctl.skill_pin_tip(str(td))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "tenth-cousins", tip["hdr"], str(td)]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "tenth-cousins")
            self.assertFalse(payload["found"])
            self.assertTrue(payload["is_only"])
            self.assertEqual(payload["entries"], [])
            self.assertEqual(payload["total"], 12)
            self.assertEqual(payload["child_hdr"], tip["hdr"])
            self.assertEqual(payload["child_height"], 11)
            self.assertEqual(payload[G9], gen["hdr"])
            self.assertEqual(payload[G9H], 0)
            self.assertFalse(payload["fetch"])
            self.assertFalse(payload["exec"])
            self.assertNotIn("body", payload)

    def test_cli_10c_missing_and_bad_hdr(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "tenth-cousins", ZERO, str(td)]
                )
            self.assertEqual(code, 1)
            payload = json.loads(buf.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["issues"], ["missing_hdr"])
        empty = juniorctl.skill_pin_tenth_cousins("", "/tmp")
        self.assertFalse(empty["ok"])
        self.assertEqual(empty["issues"], ["empty_hdr"])
        bad = juniorctl.skill_pin_tenth_cousins("tip", "/tmp")
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["issues"], ["bad_hdr"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = juniorctl.main(["juniorctl", "skill-pin", "tenth-cousins"])
        self.assertEqual(code, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["issues"], ["empty_hdr"])

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_tenth_cousins(ZERO, "/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_tenth_cousins(ZERO, "0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_tenth_cousins(ZERO, "../escape")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        empty = juniorctl.skill_pin_tenth_cousins("", "/tmp")
        self.assertFalse(empty["ok"])
        self.assertEqual(empty["issues"], ["empty_hdr"])
        bad = juniorctl.skill_pin_tenth_cousins("tip", "/tmp")
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["issues"], ["bad_hdr"])

    def test_health_lists_skill_pin_tenth_cousins(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin tenth-cousins"', src)
        self.assertIn("skill_pin_tenth_cousins", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        impl = (ROOT / "rails" / "linux" / "ctl_skillpin_tenth_cousins.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("eval(", impl)
        self.assertNotIn("exec(", impl)
        cli = (ROOT / "rails" / "linux" / "ctl_cli.py").read_text(encoding="utf-8")
        self.assertNotIn("eval(", cli)
        self.assertNotIn("exec(", cli)
        extra = (ROOT / "rails" / "linux" / "ctl_cli_t55.py").read_text(encoding="utf-8")
        self.assertIn("tenth-cousins", extra)
        self.assertIn("skill_pin_tenth_cousins", extra)
        self.assertNotIn("eval(", extra)
        self.assertNotIn("exec(", extra)
        health = juniorctl.health()
        self.assertIn("skill-pin tenth-cousins", health["cmds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
