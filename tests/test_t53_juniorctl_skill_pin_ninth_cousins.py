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
                missing["great_great_great_four_was_truncated"], ZERO
            )
