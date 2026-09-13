"""JuniorOSai wrapper. Uses ondisk.probe. Never fetches."""
from __future__ import annotations
from ports.juniorosai import JUNIOROSAI, SIZING
from ports.layer_mgr import pick_eos
from ports.ondisk import probe, ready
from ports.terraform import terraform

def run(ask: str = "juniorosai field", ram_gb: float = 8.0) -> dict:
    disk = {p.name: {"present": p.present, "path": p.path} for p in probe()}
    gguf = ready("BitNet-2B4T")
    port = pick_eos(ask, ram_gb)
    tf = terraform(ask)
    return {
        "model": JUNIOROSAI.name,
        "quant": JUNIOROSAI.quant,
        "mode": "gguf-i2s" if gguf else "kernel-ternary-1.58",
        "port": port.name,
        "disk": disk,
        "sizing": SIZING,
        "tf_ok": tf.get("ok"),
        "download": False,
    }

if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
