def status() -> dict:
    return {
        "spiffe": False,
        "mtls": False,
        "liboqs": False,
        "ml_kem": False,
        "loopback": True,
        "viz": "~/.juniorhome/gaia_mesh/tree_dense.html",
        "persist": ["tree.jsonl", "tree_dense.jsonl", "receipts.jsonl", "imager.jsonl"],
    }
