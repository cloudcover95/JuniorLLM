"""Ten live cores. Not ten new crates."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORES = [
    ("winsor_py", "junior_bitnet/winsor.py"),
    ("absmean_c", "rails/linux/absmean.c"),
    ("winsor_c", "rails/linux/winsor.c"),
    ("i2s_c", "rails/linux/i2s_pack.c"),
    ("trit_comp", "rails/linux/asahi/trit.comp"),
    ("fieldcore", "ports/fieldcore_spine.py"),
    ("flagstaff", "ports/flagstaff_balance.py"),
    ("osai", "ports/osai_suite.py"),
    ("agent_pipe", "ports/agent_pipe.py"),
    ("gguf_t3", "ports/gguf_t3.py"),
]


def check() -> dict:
    rows = [{"id": i, "path": p, "ok": (ROOT / p).is_file()} for i, p in CORES]
    return {"n": 10, "ok": all(r["ok"] for r in rows), "rust_ffi": False, "rows": rows}
