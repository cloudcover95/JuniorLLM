"""Skill-pin CLI helpers for juniorctl. Loopback only. Never fetch. Never exec."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _denied(reason: str, raw: str) -> dict:
    return {
        "ok": False,
        "issues": [reason],
        "bind": "127.0.0.1:8765",
        "root": raw,
        "skills": [],
        "count": 0,
        "chain_ok": False,
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
    }


def _guard_root(root: str | None) -> tuple[str | None, dict | None]:
    from junior_aie.skill_pin import DENY_FRAGMENTS

    raw = (root or "").strip() or str(ROOT / "grok_bot")
    low = raw.replace("\\", "/").lower()
    wildcard = ".".join(("0", "0", "0", "0"))
    if any(frag in low for frag in DENY_FRAGMENTS):
        return raw, _denied("denied_name", raw)
    if wildcard in low:
        return raw, _denied("wildcard", raw)
    if ".." in Path(raw).parts:
        return raw, _denied("path_escape", raw)
    return raw, None


def _guard_rel(text: str, raw: str) -> dict | None:
    from junior_aie.skill_pin import DENY_FRAGMENTS

    if not text:
        return _denied("empty_path", raw)
    rel_low = text.replace("\\", "/").lower()
    wildcard = ".".join(("0", "0", "0", "0"))
    if text.startswith("/") or text.startswith("~"):
        return _denied("absolute_path", raw)
    if any(frag in rel_low for frag in DENY_FRAGMENTS):
        return _denied("denied_name", raw)
    if wildcard in rel_low:
        return _denied("wildcard", raw)
    if ".." in Path(text).parts:
        return _denied("path_escape", raw)
    return None


def skill_pin_verify(root: str | None = None, skills_dir: str = "skills") -> dict:
    """T9 — verify SKILL.md pin chain under a root."""
    from junior_aie.skill_pin import SkillDenied, SkillPins, sha256_bytes

    raw, err = _guard_root(root)
    if err:
        return err
    try:
        pins = SkillPins(Path(raw))
        rels = pins.discover(skills_dir)
    except SkillDenied as exc:
        return _denied(str(exc), raw)

    issues: list[str] = []
    chain_ok = pins.verify_chain()
    if not chain_ok:
        issues.append("chain_break")

    rows: list[dict] = []
    matched = 0
    for rel in rels:
        latest = pins.latest_pin(rel)
        name = latest.name if latest else Path(rel).parent.name
        digest = None
        state = "unpinned"
        match = False
        target = pins.root / rel
        if target.is_file():
            data = target.read_bytes()
            digest = sha256_bytes(data)
        if latest is None:
            state = "unpinned"
        elif digest is None:
            state = "missing"
            issues.append("missing")
        elif digest != latest.sha256:
            state = "pin_mismatch"
            issues.append("pin_mismatch")
        else:
            state = "ok"
            match = True
            matched += 1
        rows.append(
            {
                "name": name,
                "rel": rel,
                "pinned": latest is not None,
                "match": match,
                "state": state,
                "sha256": latest.sha256 if latest else digest,
                "size": latest.size if latest else None,
            }
        )

    seen: set[str] = set()
    uniq: list[str] = []
    for item in issues:
        if item not in seen:
            seen.add(item)
            uniq.append(item)

    return {
        "ok": chain_ok and not uniq,
        "issues": uniq,
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "skills": rows,
        "count": len(rows),
        "matched": matched,
        "chain_ok": chain_ok,
        "op": "verify",
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
        "exec": False,
    }


def skill_pin_verify_one(rel: str | None = None, root: str | None = None) -> dict:
    """T10 — verify one SKILL.md REL under a root."""
    from junior_aie.skill_pin import SkillDenied, SkillPins, sha256_bytes

    raw, err = _guard_root(root)
    if err:
        return err
    text = (rel or "").strip()
    bad = _guard_rel(text, raw)
    if bad:
        return bad
    try:
        pins = SkillPins(Path(raw))
        target = pins._resolve(text)
    except SkillDenied as exc:
        return _denied(str(exc), raw)

    issues: list[str] = []
    chain_ok = pins.verify_chain()
    if not chain_ok:
        issues.append("chain_break")

    latest = pins.latest_pin(text)
    name = latest.name if latest else Path(text).parent.name
    digest = None
    state = "unpinned"
    match = False
    data = b""
    if target.is_file():
        if target.name != "SKILL.md":
            return _denied("not_skill_md", raw)
        data = target.read_bytes()
        digest = sha256_bytes(data)
    if latest is None and not target.is_file():
        return _denied("not_skill_md", raw)
    if latest is None:
        state = "unpinned"
    elif digest is None:
        state = "missing"
        issues.append("missing")
    elif digest != latest.sha256:
        state = "pin_mismatch"
        issues.append("pin_mismatch")
    else:
        state = "ok"
        match = True

    seen: set[str] = set()
    uniq: list[str] = []
    for item in issues:
        if item not in seen:
            seen.add(item)
            uniq.append(item)

    size = latest.size if latest else (len(data) if digest is not None else None)
    sha = latest.sha256 if latest else digest
    return {
        "ok": chain_ok and not uniq,
        "issues": uniq,
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "rel": text,
        "name": name,
        "op": "verify-one",
        "sha256": sha,
        "size": size,
        "skills": [
            {
                "name": name,
                "rel": text,
                "pinned": latest is not None,
                "match": match,
                "state": state,
                "sha256": sha,
                "size": size,
            }
        ],
        "count": 1,
        "matched": 1 if match else 0,
        "chain_ok": chain_ok,
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
        "exec": False,
    }


def skill_pin_tip(root: str | None = None) -> dict:
    """T11 — report SKILL_PINS.jsonl chain tip under a root. Never fetch. Never exec."""
    from junior_aie.skill_pin import ZERO, SkillDenied, SkillPins

    raw, err = _guard_root(root)
    if err:
        return err
    try:
        pins = SkillPins(Path(raw))
        rows = pins._rows()
    except SkillDenied as exc:
        return _denied(str(exc), raw)

    issues: list[str] = []
    chain_ok = pins.verify_chain()
    if not chain_ok:
        issues.append("chain_break")
    last = rows[-1] if rows else None
    return {
        "ok": chain_ok,
        "issues": issues,
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "op": "tip",
        "tip": pins.tip(),
        "hdr": last.hdr if last else ZERO,
        "height": last.height if last is not None else 0,
        "count": len(rows),
        "name": last.name if last else "",
        "rel": last.rel if last else "",
        "last_op": last.op if last else "",
        "sha256": last.sha256 if last else ZERO,
        "skills": [],
        "chain_ok": chain_ok,
        "empty": last is None,
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
        "exec": False,
    }


def skill_pin_list(root: str | None = None, skills_dir: str = "skills") -> dict:
    """T6 — list SKILL.md pins under a root."""
    from junior_aie.skill_pin import SkillDenied, SkillPins

    raw, err = _guard_root(root)
    if err:
        return err
    try:
        pins = SkillPins(Path(raw))
        rels = pins.discover(skills_dir)
    except SkillDenied as exc:
        return _denied(str(exc), raw)

    rows: list[dict] = []
    for rel in rels:
        latest = pins.latest_pin(rel)
        name = latest.name if latest else Path(rel).parent.name
        rows.append(
            {
                "name": name,
                "rel": rel,
                "pinned": latest is not None,
                "sha256": latest.sha256 if latest else None,
                "size": latest.size if latest else None,
            }
        )
    return {
        "ok": True,
        "issues": [],
        "bind": "127.0.0.1:8765",
        "root": str(Path(raw).resolve()),
        "skills": rows,
        "count": len(rows),
        "chain_ok": pins.verify_chain(),
        "docker_socket": False,
        "privileged": False,
        "download": False,
        "fetch": False,
    }
