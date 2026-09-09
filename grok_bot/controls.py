"""Machine-readable Grok Bot allow/deny + port hint."""
from __future__ import annotations

from ports.registry import pick

ALLOWED_REPOS = (
    "JuniorLLM",
    "JuniorHome",
    "JuniorClimbs",
    "AGI_SDK",
    "JuniorPiPython",
    "JuniorCoach",
    "JuniorStock",
    "JuniorAGI_SDK",
    "BitNet-mlx",
    "JuniorQuant",
    "JuniorFetch",
    "JuniorMemSys-Suite",
)

DENY_ACTIONS = (
    "delete_repo",
    "force_push",
    "private_land_public_pin",
    "gpt6_astra_client",
    "full_kimi_1_5tb",
    "bind_0_0_0_0_unauth",
)

ONE_TASK = (
    "Complete building all scripting, folders, and files of the JuniorCloud "
    "ecosystem toward a complete local software stack, with a roadmap to a "
    "custom BitNet Linux OS."
)


def allowed_repo(name: str) -> bool:
    return name in ALLOWED_REPOS


def deny(action: str) -> bool:
    return action in DENY_ACTIONS


def port_for(job: str, ram_gb: float = 8.0) -> str:
    return pick(job, ram_gb).name
