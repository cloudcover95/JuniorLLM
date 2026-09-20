"""T25 — juniorctl skill-pin count is loopback-only and never fetches."""
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


class T25JuniorctlSkillPinCountTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_count_empty_then_pin_then_load(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            missing = juniorctl.skill_pin_count(str(td))
            self.assertTrue(missing["ok"], missing)
            self.assertEqual(missing["bind"], "127.0.0.1:8765")
            self.assertEqual(missing["op"], "count")
            self.assertEqual(missing["issues"], [])
            self.assertTrue(missing["empty"])
            self.assertFalse(missing["found"])
            self.assertEqual(missing["hdr"], ZERO)
            self.assertEqual(missing["prev"], ZERO)
            self.assertEqual(missing["sha256"], ZERO)
            self.assertEqual(missing["genesis"], ZERO)
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
            self.assertNotIn("entries", missing)
            pinned = juniorctl.skill_pin_pin(rel, str(td))
            self.assertTrue(pinned["ok"], pinned)
            only = juniorctl.skill_pin_count(str(td))
            self.assertTrue(only["ok"], only)
            self.assertFalse(only["empty"])
            self.assertTrue(only["found"])
            self.assertEqual(only["last_op"], "pin")
            self.assertEqual(only["rel"], rel)
            self.assertEqual(only["name"], "plan-slice")
            self.assertEqual(only["height"], 0)
            self.assertEqual(only["count"], 1)
            self.assertEqual(only["total"], 1)
            self.assertEqual(only["prev"], ZERO)
            self.assertEqual(only["sha256"], pinned["sha256"])
            self.assertEqual(len(only["tip"] or ""), 64)
            self.assertEqual(only["tip"], only["hdr"])
            self.assertEqual(only["genesis"], only["hdr"])
            self.assertNotEqual(only["tip"], ZERO)
            self.assertTrue(only["chain_ok"])
            self.assertNotIn("entries", only)
            genesis = juniorctl.skill_pin_first(str(td))
            self.assertEqual(only["hdr"], genesis["hdr"])
            loaded = juniorctl.skill_pin_load(rel, str(td))
            self.assertTrue(loaded["ok"], loaded)
            tip1 = juniorctl.skill_pin_tip(str(td))
            self.assertTrue(tip1["ok"], tip1)
            after = juniorctl.skill_pin_count(str(td))
            self.assertTrue(after["ok"], after)
            self.assertEqual(after["last_op"], "load")
            self.assertEqual(after["count"], 2)
            self.assertEqual(after["total"], 2)
            self.assertEqual(after["height"], 1)
            self.assertEqual(after["rel"], rel)
            self.assertEqual(after["name"], "plan-slice")
            self.assertEqual(after["sha256"], pinned["sha256"])
            self.assertEqual(after["tip"], tip1["hdr"])
            self.assertEqual(after["genesis"], genesis["hdr"])
            self.assertNotEqual(after["tip"], after["genesis"])
            self.assertNotIn("entries", after)
            blob = json.dumps(after)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)
            self.assertNotIn("# Plan slice", blob)
            self.assertNotIn("body", blob)
            self.assertNotIn("entries", blob)

    def test_cli_skill_pin_count_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            juniorctl.skill_pin_pin(rel, str(td))
            juniorctl.skill_pin_load(rel, str(td))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(["juniorctl", "skill-pin", "count", str(td)])
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "count")
            self.assertEqual(payload["last_op"], "load")
            self.assertEqual(payload["height"], 1)
            self.assertEqual(payload["count"], 2)
            self.assertEqual(payload["total"], 2)
            self.assertEqual(len(payload["hdr"] or ""), 64)
            self.assertNotIn("entries", payload)

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_count("/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_count("0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_count("../escape")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        pat = juniorctl.skill_pin_count("github_pat_secret")
        self.assertFalse(pat["ok"])

    def test_health_lists_skill_pin_count(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin count"', src)
        self.assertIn("skill_pin_count", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("docker.sock", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)
        health = juniorctl.health()
        self.assertIn("skill-pin count", health["cmds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
