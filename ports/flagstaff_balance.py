"""Checks before a note becomes a node."""
from __future__ import annotations

import math

from ports.layer_mgr import pick_eos
from ports.terraform import terraform

AREAS = {
    "home", "vault", "stock", "cad", "omega", "van", "os", "llm", "app",
    "asahi", "cuda", "arm", "mlx", "cpu",
    "theory", "memsys", "engr",
    "flagstaff", "climbs", "stonefield", "xanadu", "golden", "boulder",
}

HINTS = (
    ("app", ("pyside", "gui app", "sdk module", "app development", "feature flag")),
    ("asahi", ("asahi", "agx")),
    ("cuda", ("cuda", "triton", "nvidia")),
    ("mlx", ("mlx", "metal", "m4")),
    ("arm", ("aarch64", "raspberry", "arm64")),
    ("cpu", ("x86", "i2sd pack")),
    ("cad", ("dxf", "dwg", "drawing", "omega", "height", "extrude")),
    ("engr", ("engrtools", "redline", "title block")),
    ("memsys", ("memsys", "palace", "seal")),
    ("theory", ("credit union", "zk receipt", "theorycu")),
    ("stock", ("ticker", "node", "book", "etf", "fill")),
    ("os", ("junioros", "vmlinuz", "llama")),
    ("van", ("victron", "mppt", "transit")),
    ("climbs", ("crimp", "beta", "v4", "v5", "boulder", "xanadu")),
    ("flagstaff", ("flagstaff",)),
    ("llm", ("llm", "terraform", "prompt")),
    ("home", ("home", "vault")),
)


def guess(note: str) -> str:
    t = (note or "").lower()
    for area, keys in HINTS:
        if any(k in t for k in keys):
            return area
    return "home"


def check(note: str, *, area: str = "auto", consent: bool = True, private: bool = False) -> dict:
    if area == "auto":
        area = guess(note)
    tf = terraform(note)
    port = pick_eos(note or area, 8).name
    votes = {
        "terraform_ok": bool(tf.get("ok") and tf.get("text")),
        "covenant": not (private and not consent),
        "area": area.lower() in AREAS,
        "junior_port": port.startswith("Junior"),
        "finite_y": math.isfinite(float(tf.get("fusion_y") or 0)),
        "no_bind": "0.0.0.0" not in (tf.get("text") or ""),
    }
    return {"ok": all(votes.values()), "votes": votes, "tf": tf, "port": port, "area": area.lower()}
