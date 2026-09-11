"""C7 — overlay PATH pin + systemd --user unit. Loopback only."""
from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rails.linux import juniorctl, path_pin


class C7OverlayPathUserUnitTests(unittest.TestCase):
    def test_stage_writes_profile_and_user_unit(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "prefix"
            report = path_pin.stage(dest, bind="127.0.0.1:8765")
            self.assertTrue(report["ok"], report["issues"])
            self.assertEqual(report["bind"], "127.0.0.1:8765")
            self.assertFalse(report["docker_socket"])
            self.assertFalse(report["privileged"])
            self.assertFalse(report["download"])
            profile = dest / "etc/profile.d/junioros.sh"
            unit = dest / "usr/lib/systemd/user/bitnetd.service"
            envf = dest / "etc/junioros/path.env"
            self.assertTrue(profile.is_file())
            self.assertTrue(unit.is_file())
            self.assertTrue(envf.is_file())
            ptxt = profile.read_text(encoding="utf-8")
            utxt = unit.read_text(encoding="utf-8")
            self.assertIn("/usr/local/bin", ptxt)
            self.assertIn("127.0.0.1:8765", ptxt)
            self.assertIn("WantedBy=default.target", utxt)
            self.assertIn("JUNIOR_BIND=127.0.0.1:8765", utxt)
            self.assertIn("ExecStart=/usr/local/bin/junior-bitnetd", utxt)
            self.assertNotIn("User=nobody", utxt)
            self.assertNotIn("0.0.0.0", ptxt)
            self.assertNotIn("0.0.0.0", utxt)
            self.assertNotIn("docker.sock", utxt)
            raw = json.dumps(report)
            self.assertNotIn("0.0.0.0", raw)
            self.assertNotIn("docker.sock", raw)

    def test_stage_refuses_wildcard_bind(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "bad"
            report = path_pin.stage(dest, bind="0.0.0.0:8765")
        self.assertFalse(report["ok"])
        self.assertTrue(any("bind_not_loopback" in i for i in report["issues"]))
        self.assertFalse((dest / "etc/profile.d/junioros.sh").exists())
        self.assertFalse((dest / "usr/lib/systemd/user/bitnetd.service").exists())

    def test_stage_missing_dest(self):
        report = path_pin.stage("")
        self.assertFalse(report["ok"])
        self.assertIn("missing_dest", report["issues"])

    def test_cli_path_pin(self):
        with tempfile.TemporaryDirectory() as td:
            dest = str(Path(td) / "cli")
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(["juniorctl", "path", "pin", dest])
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"], payload)
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertTrue((Path(dest) / "etc/profile.d/junioros.sh").is_file())
            self.assertTrue((Path(dest) / "usr/lib/systemd/user/bitnetd.service").is_file())

    def test_health_lists_path_pin(self):
        h = juniorctl.health()
        self.assertIn("path pin", h["cmds"])
        self.assertEqual(h["bitnetd"], "127.0.0.1:8765")

    def test_shell_installer_also_stages_path_and_user_unit(self):
        script = ROOT / "rails" / "linux" / "install-overlay.sh"
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "prefix"
            env = os.environ.copy()
            env["DEST"] = str(dest)
            env["JUNIOR_BIND"] = "127.0.0.1:8765"
            out = subprocess.check_output(["/bin/sh", str(script)], env=env, text=True)
            self.assertIn("installed JuniorOS overlay", out)
            profile = dest / "etc/profile.d/junioros.sh"
            unit = dest / "usr/lib/systemd/user/bitnetd.service"
            self.assertTrue(profile.is_file())
            self.assertTrue(unit.is_file())
            ptxt = profile.read_text(encoding="utf-8")
            utxt = unit.read_text(encoding="utf-8")
            self.assertIn("/usr/local/bin", ptxt)
            self.assertIn("127.0.0.1:8765", ptxt)
            self.assertIn("WantedBy=default.target", utxt)
            self.assertNotIn("0.0.0.0", ptxt + utxt)
            self.assertNotIn("docker.sock", utxt)
            ctl = dest / "usr/local/bin/juniorctl"
            self.assertTrue(stat.S_IXUSR & ctl.stat().st_mode)

    def test_source_has_no_eval_exec_or_wildcard(self):
        src = (ROOT / "rails" / "linux" / "path_pin.py").read_text(encoding="utf-8")
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertIn('BIND = "127.0.0.1:8765"', src)
        self.assertIn("docker.sock", src)  # forbidden-token list only
        self.assertNotIn("GPT-6", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
