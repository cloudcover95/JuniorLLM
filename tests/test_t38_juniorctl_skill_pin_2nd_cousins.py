"""T38 — juniorctl skill-pin 2nd-cousins is loopback-only and never fetches."""
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


class T38JuniorctlSkillPin2ndCousinsTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_2nd_cousins_empty_then_four_high(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            missing = juniorctl.skill_pin_2nd_cousins(ZERO, str(td))
            self.assertFalse(missing["ok"], missing)
            self.assertEqual(missing["bind"], "127.0.0.1:8765")
            self.assertEqual(missing["op"], "2nd-cousins")
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
            self.assertEqual(missing["parent_height"], -1)
            self.assertEqual(missing["grandparent_hdr"], ZERO)
            self.assertEqual(missing["great_grandparent_hdr"], ZERO)
            self.assertEqual(missing["great_grandparent_height"], -1)
            self.assertEqual(missing["last_op"], "")
            self.assertEqual(missing["entries"], [])
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
            of_gen = juniorctl.skill_pin_2nd_cousins(tip0["hdr"], str(td))
            self.assertTrue(of_gen["ok"], of_gen)
            self.assertFalse(of_gen["empty"])
            self.assertFalse(of_gen["found"])
            self.assertTrue(of_gen["is_only"])
            self.assertTrue(of_gen["is_genesis"])
            self.assertEqual(of_gen["issues"], [])
            self.assertEqual(of_gen["hdr"], ZERO)
            self.assertEqual(of_gen["prev"], ZERO)
            self.assertEqual(of_gen["sha256"], ZERO)
            self.assertEqual(of_gen["height"], -1)
            self.assertEqual(of_gen["count"], 0)
            self.assertEqual(of_gen["total"], 1)
            self.assertEqual(of_gen["entries"], [])
            self.assertEqual(of_gen["child_hdr"], tip0["hdr"])
            self.assertEqual(of_gen["child_height"], 0)
            self.assertEqual(of_gen["child_op"], "pin")
            self.assertEqual(of_gen["parent_hdr"], ZERO)
            self.assertEqual(of_gen["parent_height"], -1)
            self.assertEqual(of_gen["grandparent_hdr"], ZERO)
            self.assertEqual(of_gen["great_grandparent_hdr"], ZERO)
            self.assertEqual(of_gen["genesis"], tip0["hdr"])
            self.assertEqual(of_gen["tip"], tip0["hdr"])

            loaded = juniorctl.skill_pin_load(rel, str(td))
            self.assertTrue(loaded["ok"], loaded)
            tip1 = juniorctl.skill_pin_tip(str(td))
            of_tip = juniorctl.skill_pin_2nd_cousins(tip1["hdr"], str(td))
            self.assertTrue(of_tip["ok"], of_tip)
            self.assertFalse(of_tip["found"])
            self.assertTrue(of_tip["is_only"])
            self.assertFalse(of_tip["is_genesis"])
            self.assertEqual(of_tip["issues"], [])
            self.assertEqual(of_tip["last_op"], "")
            self.assertEqual(of_tip["rel"], "")
            self.assertEqual(of_tip["name"], "")
            self.assertEqual(of_tip["height"], -1)
            self.assertEqual(of_tip["count"], 0)
            self.assertEqual(of_tip["total"], 2)
            self.assertEqual(of_tip["prev"], ZERO)
            self.assertEqual(of_tip["sha256"], ZERO)
            self.assertEqual(of_tip["hdr"], ZERO)
            self.assertEqual(of_tip["child_hdr"], tip1["hdr"])
            self.assertEqual(of_tip["child_height"], 1)
            self.assertEqual(of_tip["child_op"], "load")
            self.assertEqual(of_tip["parent_hdr"], tip0["hdr"])
            self.assertEqual(of_tip["parent_height"], 0)
            self.assertEqual(of_tip["parent_op"], "pin")
            self.assertEqual(of_tip["grandparent_hdr"], ZERO)
            self.assertEqual(of_tip["grandparent_height"], -1)
            self.assertEqual(of_tip["great_grandparent_hdr"], ZERO)
            self.assertEqual(of_tip["tip"], tip1["hdr"])
            self.assertEqual(of_tip["genesis"], tip0["hdr"])
            self.assertTrue(of_tip["chain_ok"])
            self.assertEqual(of_tip["entries"], [])

            pinned2 = juniorctl.skill_pin_pin(rel, str(td))
            self.assertTrue(pinned2["ok"], pinned2)
            tip2 = juniorctl.skill_pin_tip(str(td))
            of_h2 = juniorctl.skill_pin_2nd_cousins(tip2["hdr"], str(td))
            self.assertTrue(of_h2["ok"], of_h2)
            self.assertFalse(of_h2["found"])
            self.assertTrue(of_h2["is_only"])
            self.assertEqual(of_h2["count"], 0)
            self.assertEqual(of_h2["total"], 3)
            self.assertEqual(of_h2["child_hdr"], tip2["hdr"])
            self.assertEqual(of_h2["child_height"], 2)
            self.assertEqual(of_h2["parent_hdr"], tip1["hdr"])
            self.assertEqual(of_h2["parent_height"], 1)
            self.assertEqual(of_h2["grandparent_hdr"], tip0["hdr"])
            self.assertEqual(of_h2["grandparent_height"], 0)
            self.assertEqual(of_h2["grandparent_op"], "pin")
            self.assertEqual(of_h2["great_grandparent_hdr"], ZERO)
            self.assertEqual(of_h2["great_grandparent_height"], -1)

            loaded2 = juniorctl.skill_pin_load(rel, str(td))
            self.assertTrue(loaded2["ok"], loaded2)
            tip3 = juniorctl.skill_pin_tip(str(td))
            of_h3 = juniorctl.skill_pin_2nd_cousins(tip3["hdr"], str(td))
            self.assertTrue(of_h3["ok"], of_h3)
            self.assertFalse(of_h3["found"])
            self.assertTrue(of_h3["is_only"])
            self.assertEqual(of_h3["issues"], [])
            self.assertEqual(of_h3["count"], 0)
            self.assertEqual(of_h3["total"], 4)
            self.assertEqual(of_h3["height"], -1)
            self.assertEqual(of_h3["last_op"], "")
            self.assertEqual(of_h3["hdr"], ZERO)
            self.assertEqual(of_h3["entries"], [])
            self.assertEqual(of_h3["child_hdr"], tip3["hdr"])
            self.assertEqual(of_h3["child_height"], 3)
            self.assertEqual(of_h3["child_op"], "load")
            self.assertEqual(of_h3["parent_hdr"], tip2["hdr"])
            self.assertEqual(of_h3["parent_height"], 2)
            self.assertEqual(of_h3["parent_op"], "pin")
            self.assertEqual(of_h3["grandparent_hdr"], tip1["hdr"])
            self.assertEqual(of_h3["grandparent_height"], 1)
            self.assertEqual(of_h3["grandparent_op"], "load")
            self.assertEqual(of_h3["great_grandparent_hdr"], tip0["hdr"])
            self.assertEqual(of_h3["great_grandparent_height"], 0)
            self.assertEqual(of_h3["great_grandparent_op"], "pin")
            self.assertEqual(of_h3["tip"], tip3["hdr"])
            self.assertEqual(of_h3["genesis"], tip0["hdr"])

            still_gen = juniorctl.skill_pin_2nd_cousins(tip0["hdr"], str(td))
            self.assertTrue(still_gen["ok"], still_gen)
            self.assertTrue(still_gen["is_genesis"])
            self.assertTrue(still_gen["is_only"])
            self.assertFalse(still_gen["found"])
            self.assertEqual(still_gen["child_hdr"], tip0["hdr"])
            self.assertEqual(still_gen["total"], 4)
            self.assertEqual(still_gen["tip"], tip3["hdr"])
            self.assertEqual(still_gen["genesis"], tip0["hdr"])

            firsts = juniorctl.skill_pin_cousins(tip3["hdr"], str(td))
            self.assertTrue(firsts["is_only"])
            self.assertFalse(firsts["found"])

            ghost = "a" * 64
            absent = juniorctl.skill_pin_2nd_cousins(ghost, str(td))
            self.assertFalse(absent["ok"])
            self.assertEqual(absent["issues"], ["missing_hdr"])
            self.assertFalse(absent["found"])
            self.assertEqual(absent["child_hdr"], ghost)
            blob = json.dumps(of_h3)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)
            self.assertNotIn("# Plan slice", blob)
            self.assertNotIn("body", blob)

    def test_cli_skill_pin_2nd_cousins_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            juniorctl.skill_pin_pin(rel, str(td))
            juniorctl.skill_pin_load(rel, str(td))
            juniorctl.skill_pin_pin(rel, str(td))
            juniorctl.skill_pin_load(rel, str(td))
            gen = juniorctl.skill_pin_genesis(str(td))
            tip = juniorctl.skill_pin_tip(str(td))
            parent = juniorctl.skill_pin_parent(tip["hdr"], str(td))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "2nd-cousins", tip["hdr"], str(td)]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "2nd-cousins")
            self.assertEqual(payload["last_op"], "")
            self.assertEqual(payload["height"], -1)
            self.assertEqual(payload["count"], 0)
            self.assertEqual(payload["total"], 4)
            self.assertEqual(payload["prev"], ZERO)
            self.assertEqual(payload["hdr"], ZERO)
            self.assertFalse(payload["found"])
            self.assertTrue(payload["is_only"])
            self.assertFalse(payload["is_genesis"])
            self.assertEqual(payload["child_hdr"], tip["hdr"])
            self.assertEqual(payload["child_height"], 3)
            self.assertEqual(payload["parent_hdr"], parent["hdr"])
            self.assertEqual(payload["parent_height"], 2)
            self.assertEqual(payload["grandparent_height"], 1)
            self.assertEqual(payload["great_grandparent_hdr"], gen["hdr"])
            self.assertEqual(payload["great_grandparent_height"], 0)
            self.assertEqual(payload["entries"], [])

    def test_cli_2nd_cousins_missing_and_bad_hdr(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "2nd-cousins", ZERO, str(td)]
                )
            self.assertEqual(code, 1)
            payload = json.loads(buf.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["issues"], ["missing_hdr"])
        empty = juniorctl.skill_pin_2nd_cousins("", "/tmp")
        self.assertFalse(empty["ok"])
        self.assertEqual(empty["issues"], ["empty_hdr"])
        bad = juniorctl.skill_pin_2nd_cousins("tip", "/tmp")
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["issues"], ["bad_hdr"])
        short = juniorctl.skill_pin_2nd_cousins("ab", "/tmp")
        self.assertFalse(short["ok"])
        self.assertEqual(short["issues"], ["bad_hdr"])
        heightish = juniorctl.skill_pin_2nd_cousins("0", "/tmp")
        self.assertFalse(heightish["ok"])
        self.assertEqual(heightish["issues"], ["bad_hdr"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = juniorctl.main(["juniorctl", "skill-pin", "2nd-cousins"])
        self.assertEqual(code, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["issues"], ["empty_hdr"])

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_2nd_cousins(ZERO, "/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_2nd_cousins(ZERO, "0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_2nd_cousins(ZERO, "../escape")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        pat = juniorctl.skill_pin_2nd_cousins(ZERO, "github_pat_secret")
        self.assertFalse(pat["ok"])

    def test_health_lists_skill_pin_2nd_cousins(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin 2nd-cousins"', src)
        self.assertIn("skill_pin_2nd_cousins", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("docker.sock", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)
        impl = (ROOT / "rails" / "linux" / "ctl_skillpin_2nd_cousins.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("eval(", impl)
        self.assertNotIn("exec(", impl)
        health = juniorctl.health()
        self.assertIn("skill-pin 2nd-cousins", health["cmds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
