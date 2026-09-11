"""C6 — overlay installer stages the C4 rootless OCI bundle. Loopback only."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rails.linux import juniorctl
from rails.linux.oci import install_bundle, rootless


class C6OverlayOciBundleTests(unittest.TestCase):
    def test_stage_copies_bundle_and_validates(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "prefix"
            report = install_bundle.stage(dest, bind="127.0.0.1:8765")
            self.assertTrue(report["ok"], report["issues"])
            self.assertEqual(report["bind"], "127.0.0.1:8765")
            self.assertTrue(report["rootless"])
            self.assertFalse(report["docker_socket"])
            self.assertFalse(report["privileged"])
            self.assertFalse(report["download"])
            cfg = dest / "usr/lib/junioros/oci/config.json"
            self.assertTrue(cfg.is_file())
            self.assertTrue((dest / "usr/lib/junioros/oci/seccomp-bitnetd.json").is_file())
            shipped = rootless.validate(json.loads(cfg.read_text(encoding="utf-8")))
            self.assertTrue(shipped["ok"], shipped["issues"])
            raw = json.dumps(report)
            self.assertNotIn("0.0.0.0", raw)
            self.assertNotIn("docker.sock", raw)

    def test_stage_refuses_wildcard_bind(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "bad"
            report = install_bundle.stage(dest, bind="0.0.0.0:8765")
        self.assertFalse(report["ok"])
        self.assertTrue(any("bind_not_loopback" in i for i in report["issues"]))
        self.assertFalse((dest / "usr/lib/junioros/oci/config.json").exists())

    def test_stage_missing_dest(self):
        report = install_bundle.stage("")
        self.assertFalse(report["ok"])
        self.assertIn("missing_dest", report["issues"])

    def test_cli_oci_install(self):
        with tempfile.TemporaryDirectory() as td:
            dest = str(Path(td) / "cli")
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = juniorctl.main(["juniorctl", "oci", "install", dest])
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"], payload)
            self.assertEqual(payload["bind"], "127.0.0.1:8765")
            self.assertTrue((Path(dest) / "usr/lib/junioros/oci/config.json").is_file())

    def test_health_lists_oci_install(self):
        h = juniorctl.health()
        self.assertIn("oci install", h["cmds"])
        self.assertEqual(h["bitnetd"], "127.0.0.1:8765")

    def test_shell_installer_also_stages_bundle(self):
        import stat
        import subprocess

        script = ROOT / "rails" / "linux" / "install-overlay.sh"
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "prefix"
            env = os.environ.copy()
            env["DEST"] = str(dest)
            env["JUNIOR_BIND"] = "127.0.0.1:8765"
            out = subprocess.check_output(["/bin/sh", str(script)], env=env, text=True)
            self.assertIn("installed JuniorOS overlay", out)
            cfg = dest / "usr/lib/junioros/oci/config.json"
            self.assertTrue(cfg.is_file())
            text = cfg.read_text(encoding="utf-8")
            self.assertIn("127.0.0.1:8765", text)
            self.assertNotIn("0.0.0.0", text)
            self.assertNotIn("docker.sock", text)
            ctl = dest / "usr/local/bin/juniorctl"
            self.assertTrue(stat.S_IXUSR & ctl.stat().st_mode)

    def test_source_has_no_eval_exec_or_sock(self):
        src = (ROOT / "rails" / "linux" / "oci" / "install_bundle.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertIn('BIND = "127.0.0.1:8765"', src)
        self.assertIn("docker.sock", src)  # forbidden-token list only
        self.assertNotIn("GPT-6", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
