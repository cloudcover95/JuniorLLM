from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grok_bot.controls import ALLOWED_REPOS, ONE_TASK, allowed_repo, deny, port_for


class ControlTests(unittest.TestCase):
    def test_llm_home_allowed(self):
        self.assertTrue(allowed_repo("JuniorLLM"))

    def test_deny_force_push(self):
        self.assertTrue(deny("force_push"))
        self.assertFalse(deny("additive_commit"))

    def test_one_task(self):
        self.assertIn("Linux OS", ONE_TASK)

    def test_astra_job(self):
        self.assertEqual(port_for("astra durable checkpoint"), "JuniorAstra")


if __name__ == "__main__":
    unittest.main(verbosity=2)
