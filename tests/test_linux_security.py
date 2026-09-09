from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rails.linux.juniorctl import security

LINUX = ROOT / "rails" / "linux"


class OverlaySecurityTests(unittest.TestCase):
    def test_unit_hardened(self):
        r = security()
        self.assertTrue(r["unit_ok"], r["missing"])
        self.assertTrue(r["seccomp"])
        self.assertFalse(r["privileged"])
        self.assertFalse(r["docker_socket"])

    def test_seccomp_default_deny(self):
        raw = json.loads((LINUX / "seccomp-bitnetd.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["defaultAction"], "SCMP_ACT_ERRNO")

    def test_installer_exists(self):
        self.assertTrue((LINUX / "install-overlay.sh").is_file())
        self.assertTrue((LINUX / "CONTAINER_SECURITY.md").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
