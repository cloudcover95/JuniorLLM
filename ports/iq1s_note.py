"""Map families. Do not decode IQ1_S grids here."""
FAMILIES = {
    "junior_note": {"bpw": 1.58, "code": {0: 0, 1: 1, -1: 2}, "needs_imatrix": False},
    "bitnet_i2s": {"bpw": 1.58, "code": {-1: 0, 0: 1, 1: 2}, "block": 128},
    "iq1_s": {"bpw": 1.56, "grid": 2048, "needs_imatrix": True},
    "tq1_0": {"bpw": 1.69, "gguf": True},
}
