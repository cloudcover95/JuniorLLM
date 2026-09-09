from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INSTALL = ROOT / "rails" / "linux" / "install-overlay.sh"


class LinuxOverlayTests(unittest.TestCase):
    def test_installer_present(self) -> None:
        self.assertTrue(INSTALL.is_file())

    def test_dry_run_needs_dest(self) -> None:
        r = subprocess.run(
            ["sh", str(INSTALL), "--dry-run"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 2)

    def test_refuse_non_loopback(self) -> None:
        env = os.environ.copy()
        env["JUNIOR_BIND"] = "0.0.0.0:8765"
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run(
                ["sh", str(INSTALL), "--dest", td, "--dry-run"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                env=env,
            )
        self.assertEqual(r.returncode, 3)
        self.assertIn("non-loopback", r.stderr)

    def test_copy_os_release_and_unit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "overlay"
            r = subprocess.run(
                ["sh", str(INSTALL), "--dest", str(dest)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            rel = dest / "etc" / "os-release.junior"
            unit = dest / "etc" / "systemd" / "system" / "bitnetd.service"
            daemon = dest / "usr" / "local" / "bin" / "junior-bitnetd"
            ctl = dest / "usr" / "local" / "bin" / "juniorctl"
            self.assertTrue(rel.is_file())
            self.assertTrue(unit.is_file())
            self.assertTrue(daemon.is_file())
            self.assertTrue(ctl.is_file())
            text = rel.read_text(encoding="utf-8") + unit.read_text(encoding="utf-8")
            self.assertIn("127.0.0.1:8765", text)
            self.assertNotIn("0.0.0.0", text)
            self.assertIn("ID=junioros", rel.read_text(encoding="utf-8"))
            mode = stat.S_IMODE(daemon.stat().st_mode)
            self.assertTrue(mode & stat.S_IXUSR)

    def test_does_not_clobber_host_os_release(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "overlay"
            host = dest / "etc" / "os-release"
            host.parent.mkdir(parents=True)
            host.write_text("NAME=host\n", encoding="utf-8")
            r = subprocess.run(
                ["sh", str(INSTALL), "--dest", str(dest)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(host.read_text(encoding="utf-8"), "NAME=host\n")
            self.assertTrue((dest / "etc" / "os-release.junior").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
