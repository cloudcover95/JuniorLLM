"""ROS 2 DDS status. Operator box, not T0."""
from __future__ import annotations


def status() -> dict:
    return {
        "rmw": None,
        "dds": False,
        "cyclone": False,
        "fast_dds": False,
        "zenoh": False,
        "multicast_discovery": False,
        "cmd_vel": False,
        "fire": False,
        "onboard_when": ["robot-product", "pinned-rmw", "no-multicast-on-felt"],
        "map": {
            "topic": "note-kind",
            "sample": "i2s-pack",
            "reliable_qos": "flagstaff-vote",
            "transient_local": "jsonl-append",
        },
    }
