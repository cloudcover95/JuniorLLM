"""T37 — juniorctl skill-pin great-great-grandchildren is loopback-only and never fetches."""
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


class T37JuniorctlSkillPinGreatGreatGrandchildrenTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_great_great_grandchildren_empty_then_five_high(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            missing = juniorctl.skill_pin_great_great_grandchildren(ZERO, str(td))
            self.assertFalse(missing["ok"], missing)
            self.assertEqual(missing["bind"], "127.0.0.1:8765")
            self.assertEqual(missing["op"], "great-great-grandchildren")
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
            self.assertEqual(missing["great_great_grandparent_hdr"], ZERO)
            self.assertEqual(missing["great_great_grandparent_height"], -1)
            self.assertEqual(missing["great_grandparent_hdr"], ZERO)
            self.assertEqual(missing["grandparent_hdr"], ZERO)
            self.assertEqual(missing["parent_hdr"], ZERO)
            self.assertEqual(missing["last_op"], "")
            self.assertEqual(missing["entries"], [])
            self.assertFalse(missing["fetch"])
            self.assertFalse(missing["download"])
            self.assertFalse(missing["docker_socket"])
            self.assertFalse(missing["privileged"])
            self.assertFalse(missing["exec"])

            pinned = juniorctl.skill_pin_pin(rel, str(td))
            self.assertTrue(pinned["ok"], pinned)
            tip0 = juniorctl.skill_pin_tip(str(td))
            self.assertTrue(tip0["ok"], tip0)
            of_gen = juniorctl.skill_pin_great_great_grandchildren(tip0["hdr"], str(td))
            self.assertTrue(of_gen["ok"], of_gen)
            self.assertFalse(of_gen["empty"])
            self.assertFalse(of_gen["found"])
            self.assertTrue(of_gen["is_only"])
            self.assertTrue(of_gen["is_genesis"])
            self.assertEqual(of_gen["issues"], [])
            self.assertEqual(of_gen["hdr"], ZERO)
            self.assertEqual(of_gen["height"], -1)
            self.assertEqual(of_gen["count"], 0)
            self.assertEqual(of_gen["total"], 1)
            self.assertEqual(of_gen["entries"], [])
            self.assertEqual(of_gen["great_great_grandparent_hdr"], tip0["hdr"])
            self.assertEqual(of_gen["great_great_grandparent_height"], 0)
            self.assertEqual(of_gen["great_great_grandparent_op"], "pin")
            self.assertEqual(of_gen["great_grandparent_hdr"], ZERO)
            self.assertEqual(of_gen["parent_hdr"], ZERO)
            self.assertEqual(of_gen["genesis"], tip0["hdr"])
            self.assertEqual(of_gen["tip"], tip0["hdr"])

            loaded = juniorctl.skill_pin_load(rel, str(td))
            self.assertTrue(loaded["ok"], loaded)
            tip1 = juniorctl.skill_pin_tip(str(td))
            pinned2 = juniorctl.skill_pin_pin(rel, str(td))
            self.assertTrue(pinned2["ok"], pinned2)
            tip2 = juniorctl.skill_pin_tip(str(td))
            loaded2 = juniorctl.skill_pin_load(rel, str(td))
            self.assertTrue(loaded2["ok"], loaded2)
            tip3 = juniorctl.skill_pin_tip(str(td))

            of_four = juniorctl.skill_pin_great_great_grandchildren(tip0["hdr"], str(td))
            self.assertTrue(of_four["ok"], of_four)
            self.assertFalse(of_four["found"])
            self.assertTrue(of_four["is_only"])
            self.assertTrue(of_four["is_genesis"])
            self.assertEqual(of_four["issues"], [])
            self.assertEqual(of_four["height"], -1)
            self.assertEqual(of_four["count"], 0)
            self.assertEqual(of_four["total"], 4)
            self.assertEqual(of_four["entries"], [])
            self.assertEqual(of_four["great_great_grandparent_hdr"], tip0["hdr"])

            of_mid = juniorctl.skill_pin_great_great_grandchildren(tip1["hdr"], str(td))
            self.assertTrue(of_mid["ok"], of_mid)
            self.assertFalse(of_mid["found"])
            self.assertTrue(of_mid["is_only"])
            self.assertFalse(of_mid["is_genesis"])
            self.assertEqual(of_mid["count"], 0)
            self.assertEqual(of_mid["height"], -1)

            of_tip3 = juniorctl.skill_pin_great_great_grandchildren(tip3["hdr"], str(td))
            self.assertTrue(of_tip3["ok"], of_tip3)
            self.assertFalse(of_tip3["found"])
            self.assertTrue(of_tip3["is_only"])
            self.assertEqual(of_tip3["count"], 0)

            pinned3 = juniorctl.skill_pin_pin(rel, str(td))
            self.assertTrue(pinned3["ok"], pinned3)
            tip4 = juniorctl.skill_pin_tip(str(td))
            of_gen5 = juniorctl.skill_pin_great_great_grandchildren(tip0["hdr"], str(td))
            self.assertTrue(of_gen5["ok"], of_gen5)
            self.assertTrue(of_gen5["found"])
            self.assertFalse(of_gen5["is_only"])
            self.assertTrue(of_gen5["is_genesis"])
            self.assertEqual(of_gen5["issues"], [])
            self.assertEqual(of_gen5["count"], 1)
            self.assertEqual(of_gen5["total"], 5)
            self.assertEqual(of_gen5["height"], 4)
            self.assertEqual(of_gen5["last_op"], "pin")
            self.assertEqual(of_gen5["hdr"], tip4["hdr"])
            self.assertEqual(of_gen5["prev"], tip3["hdr"])
            self.assertEqual(of_gen5["parent_hdr"], tip3["hdr"])
            self.assertEqual(of_gen5["parent_height"], 3)
            self.assertEqual(of_gen5["parent_op"], "load")
            self.assertEqual(of_gen5["grandparent_hdr"], tip2["hdr"])
            self.assertEqual(of_gen5["grandparent_height"], 2)
            self.assertEqual(of_gen5["grandparent_op"], "pin")
            self.assertEqual(of_gen5["great_grandparent_hdr"], tip1["hdr"])
            self.assertEqual(of_gen5["great_grandparent_height"], 1)
            self.assertEqual(of_gen5["great_grandparent_op"], "load")
            self.assertEqual(of_gen5["great_great_grandparent_hdr"], tip0["hdr"])
            self.assertEqual(of_gen5["great_great_grandparent_height"], 0)
            self.assertEqual(of_gen5["great_great_grandparent_op"], "pin")
            self.assertEqual(of_gen5["tip"], tip4["hdr"])
            self.assertEqual(of_gen5["genesis"], tip0["hdr"])
            self.assertEqual(len(of_gen5["entries"]), 1)
            self.assertEqual(of_gen5["entries"][0]["hdr"], tip4["hdr"])
            self.assertEqual(of_gen5["entries"][0]["height"], 4)
            self.assertEqual(of_gen5["entries"][0]["prev"], tip3["hdr"])
            self.assertTrue(of_gen5["chain_ok"])

            of_tip4 = juniorctl.skill_pin_great_great_grandchildren(tip4["hdr"], str(td))
            self.assertTrue(of_tip4["ok"], of_tip4)
            self.assertFalse(of_tip4["found"])
            self.assertTrue(of_tip4["is_only"])
            self.assertEqual(of_tip4["count"], 0)
            self.assertEqual(of_tip4["great_great_grandparent_hdr"], tip4["hdr"])

            ghost = "a" * 64
            absent = juniorctl.skill_pin_great_great_grandchildren(ghost, str(td))
            self.assertFalse(absent["ok"])
            self.assertEqual(absent["issues"], ["missing_hdr"])
            self.assertFalse(absent["found"])
            self.assertEqual(absent["great_great_grandparent_hdr"], ghost)
            blob = json.dumps(of_gen5)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)
            self.assertNotIn("# Plan slice", blob)
            self.assertNotIn("body", blob)

    def test_cli_skill_pin_great_great_grandchildren_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            juniorctl.skill_pin_pin(rel, str(td))
            juniorctl.skill_pin_load(rel, str(td))
            juniorctl.skill_pin_pin(rel, str(td))
            juniorctl.skill_pin_load(rel, str(td))
            juniorctl.skill_pin_pin(rel, str(td))
            gen = juniorctl.skill_pin_genesis(str(td))
            tip = juniorctl.skill_pin_tip(str(td))
            parent = juniorctl.skill_pin_parent(tip["hdr"], str(td))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "great-great-grandchildren", gen["hdr"], str(td)]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "great-great-grandchildren")
            self.assertEqual(payload["last_op"], "pin")
            self.assertEqual(payload["height"], 4)
            self.assertEqual(payload["count"], 1)
            self.assertEqual(payload["total"], 5)
            self.assertEqual(payload["prev"], parent["hdr"])
            self.assertEqual(payload["hdr"], tip["hdr"])
            self.assertTrue(payload["found"])
            self.assertFalse(payload["is_only"])
            self.assertTrue(payload["is_genesis"])
            self.assertEqual(payload["great_great_grandparent_hdr"], gen["hdr"])
            self.assertEqual(payload["great_great_grandparent_height"], 0)
            self.assertEqual(payload["parent_hdr"], parent["hdr"])
            self.assertEqual(payload["parent_height"], 3)
            self.assertEqual(len(payload["entries"]), 1)

    def test_cli_great_great_grandchildren_missing_and_bad_hdr(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "great-great-grandchildren", ZERO, str(td)]
                )
            self.assertEqual(code, 1)
            payload = json.loads(buf.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["issues"], ["missing_hdr"])
        empty = juniorctl.skill_pin_great_great_grandchildren("", "/tmp")
        self.assertFalse(empty["ok"])
        self.assertEqual(empty["issues"], ["empty_hdr"])
        bad = juniorctl.skill_pin_great_great_grandchildren("tip", "/tmp")
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["issues"], ["bad_hdr"])
        short = juniorctl.skill_pin_great_great_grandchildren("ab", "/tmp")
        self.assertFalse(short["ok"])
        self.assertEqual(short["issues"], ["bad_hdr"])
        heightish = juniorctl.skill_pin_great_great_grandchildren("0", "/tmp")
        self.assertFalse(heightish["ok"])
        self.assertEqual(heightish["issues"], ["bad_hdr"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = juniorctl.main(["juniorctl", "skill-pin", "great-great-grandchildren"])
        self.assertEqual(code, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["issues"], ["empty_hdr"])

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_great_great_grandchildren(ZERO, "/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_great_great_grandchildren(ZERO, "0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_great_great_grandchildren(ZERO, "../escape")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        pat = juniorctl.skill_pin_great_great_grandchildren(ZERO, "github_pat_secret")
        self.assertFalse(pat["ok"])

    def test_health_lists_skill_pin_great_great_grandchildren(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin great-great-grandchildren"', src)
        self.assertIn("skill_pin_great_great_grandchildren", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("docker.sock", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)
        impl = (ROOT / "rails" / "linux" / "ctl_skillpin_great_great_grandchildren.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("eval(", impl)
        self.assertNotIn("exec(", impl)
        health = juniorctl.health()
        self.assertIn("skill-pin great-great-grandchildren", health["cmds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
