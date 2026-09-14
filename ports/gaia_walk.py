"""Gaia walks to a public StoneField node. Holds → trit pack."""
from __future__ import annotations

from junior_bitnet.i2s import pack as i2s_pack
from junior_bitnet.winsor import pack as winsor_pack
from ports.flagstaff_balance import check
from ports.gaia import spine
from ports.stonefield import find, list_public


def walk(ask: str) -> dict:
    gate = check(ask or "stonefield", area="auto")
    node = find(ask) or (list_public()[0] if list_public() else None)
    if not node:
        return {"ok": False, "reason": "no-public-node", "scrape": False}
    if not node.get("public"):
        return {"ok": False, "reason": "covenant", "scrape": False}
    who = spine(ask or node["name"])
    holds = node.get("holds") or []
    xs = [float((ord(h[0]) % 13) - 6) for h in holds] or [0.2]
    q = winsor_pack(xs)
    blob = i2s_pack([int(t) for t in q.get("trit") or [0]])
    return {
        "ok": bool(gate.get("ok") and who.get("ok")),
        "who": who.get("who"),
        "port": "JuniorBitNetFieldCore",
        "node": node,
        "holds": holds,
        "routes": node.get("routes"),
        "trit": q.get("trit"),
        "gamma": q.get("gamma"),
        "i2s_hex": blob.hex(),
        "walk": ["home", node.get("area"), node.get("name")],
        "scrape": False,
        "ue5_launch": False,
        "llama_ready": False,
    }
