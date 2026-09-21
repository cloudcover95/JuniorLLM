"""T27 — juniorctl skill-pin parent is loopback-only and never fetches."""
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


class T27JuniorctlSkillPinParentTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_parent_empty_then_pin_then_load(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            missing = juniorctl.skill_pin_parent(ZERO, str(td))
            self.assertFalse(missing["ok"], missing)
            self.assertEqual(missing["bind"], "127.0.0.1:8765")
            self.assertEqual(missing["op"], "parent")
            self.assertEqual(missing["issues"], ["missing_hdr"])
            self.assertTrue(missing["empty"])
            self.assertFalse(missing["found"])
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
            self.assertEqual(missing["last_op"], "")
            self.assertFalse(missing["fetch"])
            self.assertFalse(missing["download"])
            self.assertFalse(missing["docker_socket"])
            self.assertFalse(missing["privileged"])
            self.assertFalse(missing["exec"])
            self.assertNotIn("body", missing)
            self.assertNotIn("entries", missing)
            pinned = juniorctl.skill_pin_pin(rel, str(td))
            self.assertTrue(pinned["ok"], pinned)
            tip0 = juniorctl.skill_pin_tip(str(td))
            self.assertTrue(tip0["ok"], tip0)
            genesis = juniorctl.skill_pin_genesis(str(td))
            self.assertTrue(genesis["ok"], genesis)
            of_gen = juniorctl.skill_pin_parent(tip0["hdr"], str(td))
            self.assertTrue(of_gen["ok"], of_gen)
            self.assertFalse(of_gen["empty"])
            self.assertFalse(of_gen["found"])
            self.assertTrue(of_gen["is_genesis"])
            self.assertEqual(of_gen["issues"], [])
            self.assertEqual(of_gen["hdr"], ZERO)
            self.assertEqual(of_gen["prev"], ZERO)
            self.assertEqual(of_gen["sha256"], ZERO)
            self.assertEqual(of_gen["height"], -1)
            self.assertEqual(of_gen["count"], 0)
            self.assertEqual(of_gen["total"], 1)
            self.assertEqual(of_gen["child_hdr"], tip0["hdr"])
            self.assertEqual(of_gen["child_height"], 0)
            self.assertEqual(of_gen["child_op"], "pin")
            self.assertEqual(of_gen["genesis"], tip0["hdr"])
            self.assertEqual(of_gen["tip"], tip0["hdr"])
            self.assertNotIn("entries", of_gen)
            loaded = juniorctl.skill_pin_load(rel, str(td))
            self.assertTrue(loaded["ok"], loaded)
            tip1 = juniorctl.skill_pin_tip(str(td))
            self.assertTrue(tip1["ok"], tip1)
            of_tip = juniorctl.skill_pin_parent(tip1["hdr"], str(td))
            self.assertTrue(of_tip["ok"], of_tip)
            self.assertTrue(of_tip["found"])
            self.assertFalse(of_tip["is_genesis"])
            self.assertEqual(of_tip["last_op"], "pin")
            self.assertEqual(of_tip["rel"], rel)
            self.assertEqual(of_tip["name"], "plan-slice")
            self.assertEqual(of_tip["height"], 0)
            self.assertEqual(of_tip["count"], 1)
            self.assertEqual(of_tip["total"], 2)
            self.assertEqual(of_tip["prev"], ZERO)
            self.assertEqual(of_tip["sha256"], pinned["sha256"])
            self.assertEqual(of_tip["hdr"], tip0["hdr"])
            self.assertEqual(of_tip["hdr"], genesis["hdr"])
            self.assertEqual(of_tip["child_hdr"], tip1["hdr"])
            self.assertEqual(of_tip["child_height"], 1)
            self.assertEqual(of_tip["child_op"], "load")
            self.assertEqual(of_tip["tip"], tip1["hdr"])
            self.assertNotEqual(of_tip["tip"], of_tip["hdr"])
            self.assertTrue(of_tip["chain_ok"])
            self.assertNotIn("entries", of_tip)
            still_gen = juniorctl.skill_pin_parent(tip0["hdr"], str(td))
            self.assertTrue(still_gen["ok"], still_gen)
            self.assertTrue(still_gen["is_genesis"])
            self.assertFalse(still_gen["found"])
            self.assertEqual(still_gen["child_hdr"], tip0["hdr"])
            self.assertEqual(still_gen["child_height"], 0)
            self.assertEqual(still_gen["total"], 2)
            self.assertEqual(still_gen["tip"], tip1["hdr"])
            self.assertEqual(still_gen["genesis"], tip0["hdr"])
            ghost = "a" * 64
            absent = juniorctl.skill_pin_parent(ghost, str(td))
            self.assertFalse(absent["ok"])
            self.assertEqual(absent["issues"], ["missing_hdr"])
            self.assertFalse(absent["found"])
            self.assertEqual(absent["child_hdr"], ghost)
            blob = json.dumps(of_tip)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)
            self.assertNotIn("# Plan slice", blob)
            self.assertNotIn("body", blob)
            self.assertNotIn("entries", blob)

    def test_cli_skill_pin_parent_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            juniorctl.skill_pin_pin(rel, str(td))
            juniorctl.skill_pin_load(rel, str(td))
            tip = juniorctl.skill_pin_tip(str(td))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "parent", tip["hdr"], str(td)]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "parent")
            self.assertEqual(payload["last_op"], "pin")
            self.assertEqual(payload["height"], 0)
            self.assertEqual(payload["count"], 1)
            self.assertEqual(payload["total"], 2)
            self.assertEqual(payload["prev"], ZERO)
            self.assertTrue(payload["found"])
            self.assertFalse(payload["is_genesis"])
            self.assertEqual(payload["child_hdr"], tip["hdr"])
            self.assertEqual(payload["child_height"], 1)
            self.assertNotIn("entries", payload)

    def test_cli_parent_missing_and_bad_hdr(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "parent", ZERO, str(td)]
                )
            self.assertEqual(code, 1)
            payload = json.loads(buf.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["issues"], ["missing_hdr"])
        empty = juniorctl.skill_pin_parent("", "/tmp")
        self.assertFalse(empty["ok"])
        self.assertEqual(empty["issues"], ["empty_hdr"])
        bad = juniorctl.skill_pin_parent("tip", "/tmp")
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["issues"], ["bad_hdr"])
        short = juniorctl.skill_pin_parent("ab", "/tmp")
        self.assertFalse(short["ok"])
        self.assertEqual(short["issues"], ["bad_hdr"])
        heightish = juniorctl.skill_pin_parent("0", "/tmp")
        self.assertFalse(heightish["ok"])
        self.assertEqual(heightish["issues"], ["bad_hdr"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = juniorctl.main(["juniorctl", "skill-pin", "parent"])
        self.assertEqual(code, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["issues"], ["empty_hdr"])

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_parent(ZERO, "/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_parent(ZERO, "0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_parent(ZERO, "../escape")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        pat = juniorctl.skill_pin_parent(ZERO, "github_pat_secret")
        self.assertFalse(pat["ok"])

    def test_health_lists_skill_pin_parent(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin parent"', src)
        self.assertIn("skill_pin_parent", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("docker.sock", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)
        health = juniorctl.health()
        self.assertIn("skill-pin parent", health["cmds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
