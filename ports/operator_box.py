def status() -> dict:
    return {
        "liboqs": False,
        "ml_kem_768": False,
        "spiffe": False,
        "mtls": False,
        "home_bind": "127.0.0.1",
        "onboard_when": ["two-hosts", "liboqs-present", "spire-or-manual-svid"],
    }
