"""T21 — juniorctl skill-pin first is loopback-only and never fetches."""
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


class T21JuniorctlSkillPinFirstTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_first_empty_then_pin_then_load(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            missing = juniorctl.skill_pin_first(str(td))
            self.assertTrue(missing["ok"], missing)
            self.assertEqual(missing["bind"], "127.0.0.1:8765")
            self.assertEqual(missing["op"], "first")
            self.assertEqual(missing["issues"], [])
            self.assertTrue(missing["empty"])
            self.assertFalse(missing["found"])
            self.assertEqual(missing["hdr"], ZERO)
            self.assertEqual(missing["prev"], ZERO)
            self.assertEqual(missing["sha256"], ZERO)
            self.assertEqual(missing["height"], -1)
            self.assertEqual(missing["count"], 0)
            self.assertEqual(missing["total"], 0)
            self.assertEqual(missing["tip"], ZERO)
            self.assertEqual(missing["last_op"], "")
            self.assertFalse(missing["fetch"])
            self.assertFalse(missing["download"])
            self.assertFalse(missing["docker_socket"])
            self.assertFalse(missing["privileged"])
            self.assertFalse(missing["exec"])
            self.assertNotIn("body", missing)
            pinned = juniorctl.skill_pin_pin(rel, str(td))
            self.assertTrue(pinned["ok"], pinned)
            genesis = juniorctl.skill_pin_first(str(td))
            self.assertTrue(genesis["ok"], genesis)
            self.assertFalse(genesis["empty"])
            self.assertTrue(genesis["found"])
            self.assertEqual(genesis["last_op"], "pin")
            self.assertEqual(genesis["rel"], rel)
            self.assertEqual(genesis["name"], "plan-slice")
            self.assertEqual(genesis["height"], 0)
            self.assertEqual(genesis["count"], 1)
            self.assertEqual(genesis["total"], 1)
            self.assertEqual(genesis["prev"], ZERO)
            self.assertEqual(genesis["sha256"], pinned["sha256"])
            self.assertEqual(len(genesis["tip"] or ""), 64)
            self.assertEqual(genesis["tip"], genesis["hdr"])
            self.assertNotEqual(genesis["tip"], ZERO)
            self.assertTrue(genesis["chain_ok"])
            loaded = juniorctl.skill_pin_load(rel, str(td))
            self.assertTrue(loaded["ok"], loaded)
            tip1 = juniorctl.skill_pin_tip(str(td))
            self.assertTrue(tip1["ok"], tip1)
            still = juniorctl.skill_pin_first(str(td))
            self.assertTrue(still["ok"], still)
            self.assertEqual(still["last_op"], "pin")
            self.assertEqual(still["count"], 1)
            self.assertEqual(still["total"], 2)
            self.assertEqual(still["height"], 0)
            self.assertEqual(still["rel"], rel)
            self.assertEqual(still["name"], "plan-slice")
            self.assertEqual(still["hdr"], genesis["hdr"])
            self.assertEqual(still["sha256"], pinned["sha256"])
            self.assertNotEqual(still["tip"], still["hdr"])
            self.assertEqual(still["tip"], tip1["hdr"])
            self.assertNotIn("body", still)
            blob = json.dumps(still)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)
            self.assertNotIn("# Plan slice", blob)
            self.assertNotIn("body", blob)

    def test_cli_skill_pin_first_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            juniorctl.skill_pin_pin(rel, str(td))
            juniorctl.skill_pin_load(rel, str(td))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(["juniorctl", "skill-pin", "first", str(td)])
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "first")
            self.assertEqual(payload["last_op"], "pin")
            self.assertEqual(payload["height"], 0)
            self.assertEqual(payload["total"], 2)
            self.assertEqual(len(payload["hdr"] or ""), 64)
            self.assertNotEqual(payload["tip"], payload["hdr"])

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_first("/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_first("0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_first("../escape")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        pat = juniorctl.skill_pin_first("github_pat_secret")
        self.assertFalse(pat["ok"])

    def test_health_lists_skill_pin_first(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin first"', src)
        self.assertIn("skill_pin_first", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("docker.sock", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)
        health = juniorctl.health()
        self.assertIn("skill-pin first", health["cmds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
