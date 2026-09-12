"""T5 — skill load + hash pin."""
from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junior_aie.skill_pin import CAP_BYTES, SkillDenied, SkillPins, too_big


SAMPLE = """---
name: plan-slice
description: Start of every overnight run.
---
# Plan slice

Write 5 bullets before any git write.
"""


class T5SkillPinTests(unittest.TestCase):
    def _tree(self, td: Path) -> str:
        skill = td / "skills" / "plan-slice" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SAMPLE, encoding="utf-8")
        return "skills/plan-slice/SKILL.md"

    def test_pin_then_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            pins = SkillPins(td)
            row = pins.pin(rel)
            self.assertEqual(row.op, "pin")
            self.assertEqual(row.name, "plan-slice")
            self.assertEqual(row.size, len(SAMPLE.encode()))
            body, lrow = pins.load(rel)
            self.assertEqual(body, SAMPLE)
            self.assertEqual(lrow.op, "load")
            self.assertEqual(lrow.sha256, row.sha256)
            self.assertTrue(pins.verify_chain())
            self.assertEqual(pins.discover(), [rel])

    def test_load_unpinned_and_mismatch(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            rel = self._tree(td)
            pins = SkillPins(td)
            with self.assertRaises(SkillDenied):
                pins.load(rel)
            pins.pin(rel)
            (td / rel).write_text(SAMPLE + "\nmutated\n", encoding="utf-8")
            with self.assertRaises(SkillDenied) as ctx:
                pins.load(rel)
            self.assertEqual(str(ctx.exception), "pin_mismatch")

    def test_refuse_escape_eval_docker(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            pins = SkillPins(td)
            with self.assertRaises(SkillDenied):
                pins.pin("../escape/SKILL.md")
            with self.assertRaises(SkillDenied):
                pins.load("/etc/passwd")
            bad = td / "skills" / "evil" / "SKILL.md"
            bad.parent.mkdir(parents=True)
            bad.write_text("---\nname: evil\n---\neval(user)\n", encoding="utf-8")
            with self.assertRaises(SkillDenied):
                pins.pin("skills/evil/SKILL.md")
            sock = td / "skills" / "docker.sock" / "SKILL.md"
            sock.parent.mkdir(parents=True)
            sock.write_text(SAMPLE, encoding="utf-8")
            with self.assertRaises(SkillDenied):
                pins.pin("skills/docker.sock/SKILL.md")

    def test_cap_helper_never_allocates_8gb(self):
        self.assertTrue(too_big(CAP_BYTES + 1))
        self.assertFalse(too_big(16))
        self.assertEqual(CAP_BYTES, 8 * 1024 * 1024 * 1024)

    def test_module_stdlib_and_no_eval(self):
        src = (ROOT / "junior_aie" / "skill_pin.py").read_text(encoding="utf-8")
        self.assertNotIn("eval(", src.split("DENY_BODY", 1)[0])
        self.assertNotIn("exec(", src.split("DENY_BODY", 1)[0])
        self.assertNotIn("0.0.0.0", src.split("DENY_BODY", 1)[0])
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)
        tree = ast.parse(src)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module.split(".")[0])
        self.assertTrue(
            set(imported)
            <= {"__future__", "hashlib", "json", "dataclasses", "datetime", "pathlib"}
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
