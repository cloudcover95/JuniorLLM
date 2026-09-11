from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class OverlayTests(unittest.TestCase):
    def test_stage(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "ov"
            r = subprocess.run(
                ["sh", str(ROOT / "rails/linux/debian/stage_overlay.sh"), str(dest)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue((dest / "etc/os-release.junior").is_file())
            unit = (dest / "etc/systemd/system/bitnetd.service").read_text()
            self.assertNotIn("0.0.0.0", unit)
            self.assertTrue(Path(str(dest) + ".tar.gz").is_file())
            self.assertIn("staged", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
