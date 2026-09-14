"""OSai dash agent. Any note. Field walk only if the note asks."""
from __future__ import annotations

from junior_bitnet.i2s import pack as i2s_pack
from junior_bitnet.winsor import pack as winsor_pack
from ports.flagstaff_balance import check, guess
from ports.gaia import spine
from ports.layer_mgr import pick_eos
from rails.linux.bitnet_cpp import plan as bitnet_plan
from rails.linux.llama import plan as llama_plan

FIELD = {"flagstaff", "climbs", "stonefield", "xanadu", "boulder", "golden"}


def act(note: str) -> dict:
    area = guess(note)
    gate = check(note, area=area)
    who = spine(note)
    xs = [float(ord(c) % 13 - 6) for c in (note or "home")[:32]] or [0.2]
    q = winsor_pack(xs)
    blob = i2s_pack([int(t) for t in q.get("trit") or [0]])
    port = pick_eos(note, 8)
    field = None
    if area in FIELD:
        from ports.gaia_walk import walk

        field = walk(note)
    return {
        "ok": bool(gate.get("ok") and who.get("ok")),
        "area": area,
        "port": port.name,
        "who": who.get("who"),
        "gamma": q.get("gamma"),
        "trit": q.get("trit"),
        "i2s_hex": blob.hex(),
        "field": field,
        "llama_ready": bool(llama_plan().get("ready")),
        "bitnet_cpp_ready": bool(bitnet_plan().get("ready")),
        "ue5_launch": False,
        "scrape": False,
    }
