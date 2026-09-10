"""zkML → security audit → lake → ledger."""
from __future__ import annotations

from pathlib import Path

from bitnet_pq.lake import Lake
from bitnet_pq.ledger import Ledger
from bitnet_pq.security import audit
from bitnet_pq.zkml import prove_forward, verify_forward


def run(root: Path, x: list[int] | None = None) -> dict:
    root = Path(root)
    rec = prove_forward(x or [1, 0, -1, 1] * 8, layers=2)
    if not verify_forward(rec):
        raise RuntimeError("zkml verify failed")
    report = audit(rec.proof)
    if not report.ok:
        raise RuntimeError(report.failed)
    lake = Lake(root / "lake")
    hid = lake.put(
        {
            "layers": rec.layers,
            "challenge": rec.proof.challenge.hex(),
            "trace": rec.proof.vm.trace_root.hex(),
            "scheme": rec.proof.params.scheme,
            "secure": rec.proof.secure,
        }
    )
    led = Ledger(root / "ledger.jsonl")
    row = led.append(hid, rec.proof.challenge.hex())
    return {
        "lake_id": hid,
        "height": row.height,
        "hdr": row.hdr,
        "ledger_ok": led.verify_chain(),
        "audit_ok": report.ok,
        "secure": rec.proof.secure,
    }
