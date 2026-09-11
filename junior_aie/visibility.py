"""D6 — gym_internal notes stay off the public ask path.

Public / open-space visibility must not pack, cache, or echo gym-only
notes. gym_internal (and gym) visibility still sees them. Stdlib only.
Never fetch. Loopback policy is unchanged.
"""
from __future__ import annotations

GYM_MARKERS = (
    "gym_internal",
    "indoor gym field",
    "not a public crag",
)

# Align with rails/swift JuniorGuardrail public tenures.
PUBLIC_VIS = frozenset(
    {"public", "usfs", "blm", "nps", "osmp", "state", "county", ""}
)
GYM_VIS = frozenset({"gym_internal", "gym"})

REDACTED = "[gym_internal withheld]"


def normalize_visibility(visibility: str | None) -> str:
    v = (visibility or "public").strip().lower()
    return v or "public"


def is_public(visibility: str | None) -> bool:
    return normalize_visibility(visibility) not in GYM_VIS and normalize_visibility(
        visibility
    ) != "private"


def is_gym_internal_text(text: str | None) -> bool:
    blob = (text or "").lower()
    return any(m in blob for m in GYM_MARKERS)


def filter_notes(notes: list[str], visibility: str | None) -> list[str]:
    if not is_public(visibility):
        return list(notes)
    return [n for n in notes if not is_gym_internal_text(n)]


def filter_memory(
    memory: list[tuple[str, str]], visibility: str | None
) -> list[tuple[str, str]]:
    if not is_public(visibility):
        return list(memory)
    kept: list[tuple[str, str]] = []
    for key, val in memory:
        if is_gym_internal_text(key) or is_gym_internal_text(val):
            continue
        kept.append((key, val))
    return kept


def filter_hits(
    hits: list[tuple[str, float]], visibility: str | None
) -> list[tuple[str, float]]:
    if not is_public(visibility):
        return list(hits)
    return [(doc, score) for doc, score in hits if not is_gym_internal_text(doc)]


def scrub_public_text(text: str | None, visibility: str | None) -> str:
    raw = text or ""
    if not is_public(visibility):
        return raw
    lines = []
    dropped = False
    for line in raw.splitlines() or [raw]:
        if is_gym_internal_text(line):
            dropped = True
            continue
        lines.append(line)
    out = "\n".join(lines).strip()
    if dropped and not out:
        return REDACTED
    if dropped:
        return out
    if is_gym_internal_text(out):
        return REDACTED
    return out


def cache_key(query: str, visibility: str | None) -> str:
    return f"{normalize_visibility(visibility)}:{query}"
