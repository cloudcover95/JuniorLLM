"""JuniorLLM local rail port map.

Bind every service to 127.0.0.1 only. No remote binds, no model pulls.
Named ports for the quant / OS rail mesh.
"""

from __future__ import annotations

HOST: str = "127.0.0.1"

# Named local ports (loopback only)
PORTS: dict[str, int] = {
    "Home": 7101,
    "OSai": 7102,
    "FieldCore": 7103,
    "Flagstaff": 7104,
    "JuniorLLM": 7105,
}


def endpoint(name: str) -> str:
    """Return http://127.0.0.1:<port> for a named rail port."""
    if name not in PORTS:
        raise KeyError(f"unknown port name: {name!r}; expected one of {sorted(PORTS)}")
    return f"http://{HOST}:{PORTS[name]}"


def all_endpoints() -> dict[str, str]:
    """Map each named port to its loopback URL."""
    return {name: endpoint(name) for name in PORTS}


if __name__ == "__main__":
    for name, url in all_endpoints().items():
        print(f"{name}: {url}")
