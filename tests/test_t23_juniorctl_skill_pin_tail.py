"""T23 — juniorctl skill-pin tail is loopback-only and never fetches."""
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


class T23JuniorctlSkillPinTailTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_tail_empty_then_pin_then_load(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            missing = juniorctl.skill_pin_tail("8", str(td))
            self.assertTrue(missing["ok"], missing)
            self.assertEqual(missing["bind"], "127.0.0.1:8765")
            self.assertEqual(missing["op"], "tail")
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
            self.assertEqual(missing["n"], 8)
            self.assertEqual(missing["entries"], [])
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
            only = juniorctl.skill_pin_tail("2", str(td))
            self.assertTrue(only["ok"], only)
            self.assertFalse(only["empty"])
            self.assertTrue(only["found"])
            self.assertEqual(only["last_op"], "pin")
            self.assertEqual(only["rel"], rel)
            self.assertEqual(only["name"], "plan-slice")
            self.assertEqual(only["height"], 0)
            self.assertEqual(only["count"], 1)
            self.assertEqual(only["total"], 1)
            self.assertEqual(only["n"], 2)
            self.assertEqual(only["prev"], ZERO)
            self.assertEqual(only["sha256"], pinned["sha256"])
            self.assertEqual(len(only["tip"] or ""), 64)
            self.assertEqual(only["tip"], only["hdr"])
            self.assertEqual(only["genesis"], only["hdr"])
            self.assertEqual(only["entries"][0]["op"], "pin")
            self.assertEqual(only["entries"][0]["height"], 0)
            self.assertNotIn("body", only["entries"][0])
            self.assertNotEqual(only["tip"], ZERO)
            self.assertTrue(only["chain_ok"])
            genesis = juniorctl.skill_pin_first(str(td))
            self.assertEqual(only["hdr"], genesis["hdr"])
            loaded = juniorctl.skill_pin_load(rel, str(td))
            self.assertTrue(loaded["ok"], loaded)
            tip1 = juniorctl.skill_pin_tip(str(td))
            self.assertTrue(tip1["ok"], tip1)
            still = juniorctl.skill_pin_tail("2", str(td))
            self.assertTrue(still["ok"], still)
            self.assertEqual(still["last_op"], "load")
            self.assertEqual(still["count"], 2)
            self.assertEqual(still["total"], 2)
            self.assertEqual(still["n"], 2)
            self.assertEqual(still["height"], 1)
            self.assertEqual(still["rel"], rel)
            self.assertEqual(still["name"], "plan-slice")
            self.assertEqual(still["sha256"], pinned["sha256"])
            self.assertEqual(still["tip"], still["hdr"])
            self.assertEqual(still["tip"], tip1["hdr"])
            self.assertEqual(still["genesis"], genesis["hdr"])
            self.assertNotEqual(still["tip"], still["genesis"])
            self.assertEqual(still["entries"][0]["op"], "pin")
            self.assertEqual(still["entries"][0]["height"], 0)
            self.assertEqual(still["entries"][1]["op"], "load")
            self.assertEqual(still["entries"][1]["height"], 1)
            self.assertEqual(still["entries"][1]["prev"], still["entries"][0]["hdr"])
            one = juniorctl.skill_pin_tail("1", str(td))
            last = juniorctl.skill_pin_last(str(td))
            self.assertTrue(one["ok"], one)
            self.assertEqual(one["count"], 1)
            self.assertEqual(one["n"], 1)
            self.assertEqual(one["hdr"], last["hdr"])
            self.assertEqual(one["last_op"], "load")
            self.assertEqual(one["entries"][0]["op"], "load")
            defaulted = juniorctl.skill_pin_tail(None, str(td))
            self.assertEqual(defaulted["n"], 8)
            self.assertEqual(defaulted["count"], 2)
            blob = json.dumps(still)
            self.assertNotIn("0.0.0.0", blob)
            self.assertNotIn("docker.sock", blob)
            self.assertNotIn("# Plan slice", blob)
            self.assertNotIn("body", blob)

    def test_cli_skill_pin_tail_exit_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            juniorctl.skill_pin_pin(rel, str(td))
            juniorctl.skill_pin_load(rel, str(td))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "tail", "2", str(td)]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "tail")
            self.assertEqual(payload["last_op"], "load")
            self.assertEqual(payload["height"], 1)
            self.assertEqual(payload["total"], 2)
            self.assertEqual(payload["n"], 2)
            self.assertEqual(len(payload["hdr"] or ""), 64)
            self.assertEqual(payload["tip"], payload["hdr"])
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(["juniorctl", "skill-pin", "tail", str(td)])
            self.assertEqual(code, 0)
            root_only = json.loads(buf.getvalue())
            self.assertTrue(root_only["ok"])
            self.assertEqual(root_only["n"], 8)
            self.assertEqual(root_only["count"], 2)

    def test_cli_tail_bad_n(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            bad = juniorctl.skill_pin_tail("tip", str(td))
            self.assertFalse(bad["ok"])
            self.assertEqual(bad["issues"], ["bad_n"])
            zero = juniorctl.skill_pin_tail("0", str(td))
            self.assertFalse(zero["ok"])
            self.assertEqual(zero["issues"], ["bad_n"])
            neg = juniorctl.skill_pin_tail("-1", str(td))
            self.assertFalse(neg["ok"])
            self.assertEqual(neg["issues"], ["bad_n"])
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(
                    ["juniorctl", "skill-pin", "tail", "tip", str(td)]
                )
            self.assertEqual(code, 1)
            payload = json.loads(buf.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["issues"], ["bad_n"])

    def test_refuse_docker_wildcard_escape(self):
        sock = juniorctl.skill_pin_tail("2", "/var/run/docker.sock")
        self.assertFalse(sock["ok"])
        self.assertEqual(sock["issues"], ["denied_name"])
        wild = juniorctl.skill_pin_tail("2", "0.0.0.0")
        self.assertFalse(wild["ok"])
        self.assertEqual(wild["issues"], ["wildcard"])
        esc = juniorctl.skill_pin_tail("2", "../escape")
        self.assertFalse(esc["ok"])
        self.assertEqual(esc["issues"], ["path_escape"])
        pat = juniorctl.skill_pin_tail("2", "github_pat_secret")
        self.assertFalse(pat["ok"])

    def test_health_lists_skill_pin_tail(self):
        src = (ROOT / "rails" / "linux" / "juniorctl.py").read_text(encoding="utf-8")
        self.assertIn('"skill-pin tail"', src)
        self.assertIn("skill_pin_tail", src)
        self.assertIn("DENY_FRAGMENTS", src)
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("docker.sock", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)
        health = juniorctl.health()
        self.assertIn("skill-pin tail", health["cmds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
