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


class T53JuniorctlSkillPinNinthCousinsTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def _eleven(self, td: Path, rel: str) -> None:
        juniorctl.skill_pin_pin(rel, str(td))
        juniorctl.skill_pin_load(rel, str(td))
        juniorctl.skill_pin_pin(rel, str(td))
        juniorctl.skill_pin_load(rel, str(td))
        juniorctl.skill_pin_pin(rel, str(td))
        juniorctl.skill_pin_load(rel, str(td))
        juniorctl.skill_pin_pin(rel, str(td))
        juniorctl.skill_pin_load(rel, str(td))
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
                missing["great_great_great_great_great_great_great_grandparent_height"], -1
            )
            self.assertEqual(
                missing["great_great_great_great_great_great_great_great_grandparent_hdr"], ZERO
            )
            self.assertEqual(
                missing["great_great_great_great_great_great_great_great_grandparent_height"], -1
            )
            self.assertEqual(missing["entries"], [])
            self.assertFalse(missing["fetch"])
            self.assertFalse(missing["download"])
            self.assertFalse(missing["docker_socket"])
            self.assertFalse(missing["privileged"])
            self.assertFalse(missing["exec"])
            self.assertNotIn("body", missing)

            self._eleven(td, rel)
            tip = juniorctl.skill_pin_tip(str(td))
            gen = juniorctl.skill_pin_genesis(str(td))
            of_h10 = juniorctl.skill_pin_ninth_cousins(tip["hdr"], str(td))
            self.assertTrue(of_h10["ok"], of_h10)
            self.assertFalse(of_h10["found"])
            self.assertTrue(of_h10["is_only"])
            self.assertFalse(of_h10["is_genesis"])
            self.assertEqual(of_h10["issues"], [])
            self.assertEqual(of_h10["total"], 11)
            self.assertEqual(of_h10["count"], 0)
            self.assertEqual(of_h10["entries"], [])
            self.assertEqual(of_h10["child_hdr"], tip["hdr"])
            self.assertEqual(of_h10["child_height"], 10)
            self.assertEqual(
                of_h10["great_great_great_great_great_great_great_great_grandparent_hdr"], gen["hdr"]
            )
            self.assertEqual(
                of_h10["great_great_great_great_great_great_great_great_grandparent_height"], 0
            )
            self.assertEqual(of_h10["genesis"], gen["hdr"])
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
                    [
                        "juniorctl",
                        "skill-pin",
                        "ninth-cousins",
                        tip["hdr"],
                        str(td),
                    ]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertEqual(payload["op"], "ninth-cousins")
            self.assertFalse(payload["found"])
            self.assertTrue(payload["is_only"])
            self.assertEqual(payload["entries"], [])
            self.assertEqual(payload["total"], 11)
            self.assertEqual(payload["child_hdr"], tip["hdr"])
            self.assertEqual(payload["child_height"], 10)
            self.assertEqual(
                payload["great_great_great_four_placeholder"], gen["hdr"]
            )
