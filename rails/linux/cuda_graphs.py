def report() -> dict:
    return {
        "tensor_core": False,
        "cuda_graph": False,
        "wmma": False,
        "reason": "ternary note pack is add/sub; tiles and graphs do not pay off",
        "threshold_n": 100000,
    }
