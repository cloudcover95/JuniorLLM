"""T40 — juniorctl skill-pin 2nd-cousins-once-removed is loopback-only and never fetches."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
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


class T40JuniorctlSkillPin2ndCousinsOnceRemovedTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_2c1r_empty_then_five_high(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            missing = juniorctl.skill_pin_2nd_cousins_once_removed(ZERO, str(td))
            self.assertFalse(missing["ok"], missing)
            self.assertEqual(missing["bind"], "127.0.0.1:8765")
            self.assertEqual(missing["op"], "2nd-cousins-once-removed")
            self.assertEqual(missing["issues"], ["missing_hdr"])
            self.assertTrue(missing["empty"])
            self.assertFalse(missing["found"])
            self.assertFalse(missing["fetch"])
            self.assertFalse(missing["exec"])
            self.assertNotIn("body", missing)

            juniorctl.skill_pin_pin(rel, str(td))
            juniorctl.skill_pin_load(rel, str(td))
            juniorctl.skill_pin_pin(rel, str(td))
            juniorctl.skill_pin_load(rel, str(td))
            juniorctl.skill_pin_pin(rel, str(td))
            tip = juniorctl.skill_pin_tip(str(td))
            gen = juniorctl.skill_pin_genesis(str(td))
            of_h4 = juniorctl.skill_pin_2nd_cousins_once_removed(tip["hdr"], str(td))
            self.assertTrue(of_h4["ok"], of_h4)
            self.assertFalse(of_h4["found"])
            self.assertTrue(of_h4["is_only"])
            self.assertEqual(of_h4["total"], 5)
            self.assertEqual(of_h4["child_hdr"], tip["hdr"])
            self.assertEqual(of_h4["child_height"], 4)
            self.assertEqual(of_h4["great_great_grandparent_hdr"], gen["hdr"])
            self.assertEqual(of_h4["great_great_grandparent_height"], 0)
            self.assertEqual(of_h4["entries"], [])
            blob = json.dumps(of_h4)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)
            self.assertNotIn("body", blob)

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_2nd_cousins_once_removed(ZERO, "/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_2nd_cousins_once_removed(ZERO, "0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_2nd_cousins_once_removed(ZERO, "../escape")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        empty = juniorctl.skill_pin_2nd_cousins_once_removed("", "/tmp")
        self.assertFalse(empty["ok"])
        self.assertEqual(empty["issues"], ["empty_hdr"])
        bad = juniorctl.skill_pin_2nd_cousins_once_removed("tip", "/tmp")
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["issues"], ["bad_hdr"])

    def test_health_lists_skill_pin_2nd_cousins_once_removed(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin 2nd-cousins-once-removed"', src)
        self.assertIn("skill_pin_2nd_cousins_once_removed", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        impl = (ROOT / "rails" / "linux" / "ctl_skillpin_2nd_cousins_once_removed.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("eval(", impl)
        self.assertNotIn("exec(", impl)
        health = juniorctl.health()
        self.assertIn("skill-pin 2nd-cousins-once-removed", health["cmds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
