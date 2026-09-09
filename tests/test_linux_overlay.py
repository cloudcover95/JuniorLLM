"""Stdlib checks for the JuniorOS overlay installer (loopback only)."""
from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "rails" / "linux" / "install-overlay.sh"


class OverlayInstallTests(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(SCRIPT.is_file())

    def test_install_prefix_copies_units(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "prefix"
            env = os.environ.copy()
            env["DEST"] = str(dest)
            env["JUNIOR_BIND"] = "127.0.0.1:8765"
            out = subprocess.check_output(["/bin/sh", str(SCRIPT)], env=env, text=True)
            self.assertIn("installed JuniorOS overlay", out)
            release = (dest / "etc" / "os-release.junior").read_text()
            self.assertIn("ID=junioros", release)
            self.assertIn("127.0.0.1:8765", release)
            unit = (dest / "etc" / "systemd" / "system" / "bitnetd.service").read_text()
            self.assertIn("127.0.0.1:8765", unit)
            self.assertNotIn("0.0.0.0", unit)
            wrapper = dest / "usr" / "local" / "bin" / "junior-bitnetd"
            ctl = dest / "usr" / "local" / "bin" / "juniorctl"
            self.assertTrue(wrapper.is_file())
            self.assertTrue(ctl.is_file())
            self.assertTrue(stat.S_IXUSR & wrapper.stat().st_mode)
            self.assertTrue(stat.S_IXUSR & ctl.stat().st_mode)
            lib_rel = dest / "usr" / "lib" / "os-release.junior"
            self.assertTrue(lib_rel.is_file())

    def test_refuse_wildcard_bind(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "bad"
            env = os.environ.copy()
            env["DEST"] = str(dest)
            env["JUNIOR_BIND"] = "0.0.0.0:8765"
            proc = subprocess.run(
                ["/bin/sh", str(SCRIPT)],
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(proc.returncode, 3)
            self.assertIn("non-loopback", proc.stderr)
            self.assertFalse((dest / "etc" / "os-release.junior").exists())

    def test_missing_dest_usage(self):
        env = os.environ.copy()
        env.pop("DEST", None)
        proc = subprocess.run(
            ["/bin/sh", str(SCRIPT)],
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
