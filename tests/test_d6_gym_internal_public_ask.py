"""D6 — gym_internal notes must not leak on a public ask."""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junior_aie import build_framework
from junior_aie.corpus import seed
from junior_aie.visibility import REDACTED, filter_notes, is_gym_internal_text
from ports.ondisk import probe


GYM_NOTE = "JuniorHall is the indoor gym field. gym_internal visibility. Not a public crag."
GYM_MEM = ("gym_internal", "JuniorHall beta door note stays inside the gym.")
PUBLIC_NOTE = "Flagstaff Mountain Boulder Colorado. OSMP land. Popular problems."


class D6GymInternalPublicAskTests(unittest.TestCase):
    def test_seeded_public_ask_drops_gym_note(self):
        fw = build_framework()
        seed(fw.retrieval)
        r = fw.ask("JuniorHall gym field", visibility="public")
        self.assertTrue(r["ok"])
        self.assertEqual(r["visibility"], "public")
        blob = r["text"].lower()
        self.assertNotIn("gym_internal", blob)
        self.assertNotIn("indoor gym field", blob)
        self.assertNotIn("not a public crag", blob)

    def test_gym_visibility_still_sees_note(self):
        fw = build_framework()
        seed(fw.retrieval)
        r = fw.ask("JuniorHall gym field", visibility="gym_internal")
        self.assertTrue(r["ok"])
        self.assertEqual(r["visibility"], "gym_internal")
        self.assertIn("gym_internal", r["text"].lower())

    def test_memory_not_packed_on_public_ask(self):
        fw = build_framework()
        fw.retrieval.add(PUBLIC_NOTE)
        r = fw.ask(
            "Flagstaff conditions",
            memory=[GYM_MEM, ("covenant", "do not publish private-land boulders")],
            visibility="public",
        )
        self.assertTrue(r["ok"])
        self.assertNotIn("door note", r["text"].lower())
        self.assertNotIn("gym_internal", r["text"].lower())
        self.assertIn("covenant", r["text"].lower())

    def test_cache_does_not_cross_visibility(self):
        fw = build_framework()
        fw.retrieval.add(GYM_NOTE)
        gym = fw.ask("JuniorHall", visibility="gym_internal")
        pub = fw.ask("JuniorHall", visibility="public")
        self.assertFalse(pub.get("cached"))
        self.assertIn("gym_internal", gym["text"].lower())
        self.assertNotIn("gym_internal", pub["text"].lower())
        self.assertNotIn("indoor gym field", pub["text"].lower())
        self.assertNotIn("not a public crag", pub["text"].lower())

    def test_filter_notes_helper(self):
        kept = filter_notes([GYM_NOTE, PUBLIC_NOTE], "public")
        self.assertEqual(kept, [PUBLIC_NOTE])
        self.assertTrue(is_gym_internal_text(GYM_NOTE))
        self.assertFalse(is_gym_internal_text(PUBLIC_NOTE))
        self.assertEqual(REDACTED, "[gym_internal withheld]")

    def test_juniorctl_public_ask_seed_does_not_leak(self):
        from rails.linux import juniorctl

        out = juniorctl.ask("JuniorHall indoor gym")
        blob = str(out).lower()
        self.assertNotIn("gym_internal", blob)
        self.assertNotIn("indoor gym field", blob)

    def test_b2_ondisk_probe_never_downloads(self):
        ports = probe(ROOT / "missing-models")
        self.assertTrue(ports)
        self.assertTrue(all(not p.present for p in ports))
        src = (ROOT / "ports" / "ondisk.py").read_text(encoding="utf-8")
        self.assertIn("Never fetch", src)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)

    def test_new_module_stdlib_and_no_eval(self):
        src = (ROOT / "junior_aie" / "visibility.py").read_text(encoding="utf-8")
        self.assertNotIn("eval(", src)
        self.assertNotIn("exec(", src)
        self.assertNotIn("0.0.0.0", src)
        self.assertNotIn("docker.sock", src)
        tree = ast.parse(src)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module.split(".")[0])
        self.assertTrue(set(imported) <= {"__future__"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
