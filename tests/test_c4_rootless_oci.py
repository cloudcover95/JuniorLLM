"""C4 — rootless OCI bitnetd unit is loopback-only and socket-free."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rails.linux.oci import rootless


class C4RootlessOciTests(unittest.TestCase):
    def test_shipped_config_validates(self):
        cfg = rootless.load_config()
        report = rootless.validate(cfg)
        self.assertTrue(report["ok"], report["issues"])
        self.assertEqual(report["bind"], "127.0.0.1:8765")
        self.assertTrue(report["rootless"])
        self.assertFalse(report["privileged"])
        self.assertFalse(report["docker_socket"])

    def test_unit_refuses_docker_socket_and_wildcard(self):
        u = rootless.unit()
        self.assertEqual(u["kind"], "oci-rootless")
        self.assertEqual(u["bind"], "127.0.0.1:8765")
        self.assertFalse(u["docker_socket"])
        self.assertFalse(u["privileged"])
        self.assertEqual(u["network"], "none")
        self.assertEqual(u["status"], "ready")
        raw = json.dumps(u)
        self.assertNotIn("0.0.0.0", raw)
        self.assertNotIn("docker.sock", raw)

    def test_rejects_privileged_and_host_socket(self):
        cfg = rootless.load_config()
        cfg["process"]["noNewPrivileges"] = False
        cfg["process"]["capabilities"]["bounding"] = ["CAP_SYS_ADMIN"]
        cfg["mounts"].append(
            {
                "destination": "/var/run/docker.sock",
                "type": "bind",
                "source": "/var/run/docker.sock",
            }
        )
        cfg["process"]["env"] = ["JUNIOR_BIND=0.0.0.0:8765"]
        report = rootless.validate(cfg)
        self.assertFalse(report["ok"])
        joined = " ".join(report["issues"])
        self.assertIn("missing_noNewPrivileges", joined)
        self.assertTrue(
            "docker_socket_mount" in joined or "docker_socket_mentioned" in joined
        )
        self.assertTrue("wildcard" in joined or "bind_not_loopback" in joined)

    def test_bitnet_ondisk_probe_never_downloads(self):
        with tempfile.TemporaryDirectory() as td:
            info = rootless.bitnet_weights(Path(td))
        self.assertEqual(info["name"], "BitNet-2B4T")
        self.assertFalse(info["present"])
        self.assertFalse(info["download"])
        self.assertEqual(info["fallback"], "JuniorBitNetFieldCore")

    def test_source_has_no_eval_exec_or_sock_bind(self):
        src = (ROOT / "rails" / "linux" / "oci" / "rootless.py").read_text(encoding="utf-8")
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertIn('BIND = "127.0.0.1:8765"', src)
        cfg = (ROOT / "rails" / "linux" / "oci" / "config.json").read_text(encoding="utf-8")
        self.assertNotIn("docker.sock", cfg)
        self.assertNotIn("0.0.0.0", cfg)
        self.assertIn("127.0.0.1:8765", cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
