"""Map ROS2 names onto T-layers. No rmw import."""
T_LAYERS = ("T0", "T1", "T2", "T3")

MAP = {
    "node": "T2",
    "topic": "T0",
    "service": "T0",
    "action": "T2",
    "qos": "T0",
    "rmw": "operator",
    "dds": "operator",
}


def place(entity: str) -> dict:
    e = (entity or "").lower()
    layer = MAP.get(e, "unmapped")
    return {
        "entity": e,
        "layer": layer,
        "rmw": False,
        "cyclonedds": False,
        "multicast": False,
    }
