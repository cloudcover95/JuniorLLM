"""Gaia walks a public StoneField node. No blender -b. No 02_Assets."""
from __future__ import annotations

from junior_bitnet.i2s import pack as i2s_pack
from junior_bitnet.winsor import pack as winsor_pack
from ports.flagstaff_balance import check
from ports.gaia import spine
from ports.stonefield import find, list_public


def _xs(node: dict) -> list[float]:
    m = node.get("holds_matrix")
    if isinstance(m, list) and m:
        out = []
        for row in m:
            out.extend(float(x) for x in row)
        return out or [0.2]
    holds = node.get("holds") or []
    return [float((ord(h[0]) % 13) - 6) for h in holds] or [0.2]


def walk(ask: str) -> dict:
    gate = check(ask or "stonefield", area="auto")
    node = find(ask) or (list_public()[0] if list_public() else None)
    if not node or not node.get("public"):
        return {"ok": False, "reason": "covenant-or-missing", "scrape": False, "protocol": "goldend-osai-omega/1"}
    who = spine(ask or node["name"])
    q = winsor_pack(_xs(node))
    blob = i2s_pack([int(t) for t in q.get("trit") or [0]])
    return {
        "ok": bool(gate.get("ok") and who.get("ok")),
        "protocol": "goldend-osai-omega/1",
        "who": who.get("who"),
        "port": "JuniorBitNetFieldCore",
        "node": {"id": node.get("id"), "name": node.get("name"), "area": node.get("area"), "kind": node.get("kind")},
        "holds": node.get("holds"),
        "routes": node.get("routes"),
        "trit": q.get("trit"),
        "gamma": q.get("gamma"),
        "i2s_hex": blob.hex(),
        "walk": ["home", node.get("area"), node.get("name")],
        "omega": {"job": "gaia-spine", "mesh": "obj", "launch": False},
        "scrape": False,
        "ue5_launch": False,
        "blender_cli": False,
    }
