"""Map unbalanced {0,1,2} study symbols onto {-1,0,1}. No fab."""
from __future__ import annotations


def to_balanced(u: int) -> int:
    return {0: -1, 1: 0, 2: 1}.get(int(u), 0)


def to_unbalanced(b: int) -> int:
    return {-1: 0, 0: 1, 1: 2}.get(int(b), 1)


def status() -> dict:
    return {
        "fab": False,
        "pdk": False,
        "qutrit_hw": False,
        "cmos_ternary_hw": False,
        "sti_pti_nti": False,
        "alphabet_home": [-1, 0, 1],
        "alphabet_study": [0, 1, 2],
        "train": "jsonl",
    }
