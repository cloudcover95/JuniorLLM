"""CLI dispatch for juniorctl. Loopback only."""
from __future__ import annotations

import json
import sys


def run(argv: list[str], ns: dict) -> int:
    cmd = argv[1] if len(argv) > 1 else "health"
    if cmd == "health":
        print(json.dumps(ns["health"](), indent=2))
        return 0
    if cmd == "security":
        report = ns["security"]()
        print(json.dumps(report, indent=2))
        return 0 if report["unit_ok"] and report["seccomp"] else 1
    if cmd == "port" and len(argv) > 2 and argv[2] == "list":
        print(json.dumps(ns["ports"](), indent=2))
        return 0
    if cmd == "ask":
        q = " ".join(argv[2:]).strip() or "public field conditions brief"
        print(json.dumps(ns["ask"](q), indent=2))
        return 0
    if cmd == "night":
        print(json.dumps(ns["night"](), indent=2))
        return 0
    if cmd == "quant":
        print(json.dumps(ns["quant"](), indent=2, default=str))
        return 0
    if cmd == "lake":
        print(json.dumps(ns["lake"](), indent=2))
        return 0
    if cmd == "net":
        print(json.dumps(ns["net_status"](), indent=2))
        return 0
    if cmd == "oci":
        sub = argv[2] if len(argv) > 2 else "validate"
        if sub == "validate":
            report = ns["oci_validate"]()
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "install":
            dest = argv[3] if len(argv) > 3 else None
            report = ns["oci_install"](dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        print("usage: juniorctl oci validate | oci install [DEST]", file=sys.stderr)
        return 2
    if cmd == "path":
        sub = argv[2] if len(argv) > 2 else "pin"
        if sub == "pin":
            dest = argv[3] if len(argv) > 3 else None
            report = ns["path_pin"](dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        print("usage: juniorctl path pin [DEST]", file=sys.stderr)
        return 2
    if cmd == "skill-pin":
        sub = argv[2] if len(argv) > 2 else "list"
        if sub == "list":
            dest = argv[3] if len(argv) > 3 else None
            report = ns["skill_pin_list"](dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "load":
            rel = argv[3] if len(argv) > 3 else None
            dest = argv[4] if len(argv) > 4 else None
            report = ns["skill_pin_load"](rel, dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "pin":
            rel = argv[3] if len(argv) > 3 else None
            dest = argv[4] if len(argv) > 4 else None
            report = ns["skill_pin_pin"](rel, dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "verify":
            dest = argv[3] if len(argv) > 3 else None
            report = ns["skill_pin_verify"](dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "verify-one":
            rel = argv[3] if len(argv) > 3 else None
            dest = argv[4] if len(argv) > 4 else None
            report = ns["skill_pin_verify_one"](rel, dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "tip":
            dest = argv[3] if len(argv) > 3 else None
            report = ns["skill_pin_tip"](dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "log":
            dest = argv[3] if len(argv) > 3 else None
            report = ns["skill_pin_log"](dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "height":
            dest = argv[3] if len(argv) > 3 else None
            report = ns["skill_pin_height"](dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "get":
            height = argv[3] if len(argv) > 3 else None
            dest = argv[4] if len(argv) > 4 else None
            report = ns["skill_pin_get"](height, dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "at":
            hdr = argv[3] if len(argv) > 3 else None
            dest = argv[4] if len(argv) > 4 else None
            report = ns["skill_pin_at"](hdr, dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "range":
            start = argv[3] if len(argv) > 3 else None
            stop = argv[4] if len(argv) > 4 else None
            dest = argv[5] if len(argv) > 5 else None
            report = ns["skill_pin_range"](start, stop, dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "since":
            hdr = argv[3] if len(argv) > 3 else None
            dest = argv[4] if len(argv) > 4 else None
            report = ns["skill_pin_since"](hdr, dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "until":
            hdr = argv[3] if len(argv) > 3 else None
            dest = argv[4] if len(argv) > 4 else None
            report = ns["skill_pin_until"](hdr, dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "before":
            hdr = argv[3] if len(argv) > 3 else None
            dest = argv[4] if len(argv) > 4 else None
            report = ns["skill_pin_before"](hdr, dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "after":
            hdr = argv[3] if len(argv) > 3 else None
            dest = argv[4] if len(argv) > 4 else None
            report = ns["skill_pin_after"](hdr, dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        if sub == "first":
            dest = argv[3] if len(argv) > 3 else None
            report = ns["skill_pin_first"](dest)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        print(
            "usage: juniorctl skill-pin list [ROOT] | skill-pin load REL [ROOT] | skill-pin pin REL [ROOT] | skill-pin verify [ROOT] | skill-pin verify-one REL [ROOT] | skill-pin tip [ROOT] | skill-pin log [ROOT] | skill-pin height [ROOT] | skill-pin get HEIGHT [ROOT] | skill-pin at HDR [ROOT] | skill-pin range FROM TO [ROOT] | skill-pin since HDR [ROOT] | skill-pin until HDR [ROOT] | skill-pin before HDR [ROOT] | skill-pin after HDR [ROOT] | skill-pin first [ROOT]",
            file=sys.stderr,
        )
        return 2
    print(
        "usage: juniorctl health | security | port list | ask <q> | night | quant | lake | net | oci validate | oci install [DEST] | path pin [DEST] | skill-pin list [ROOT] | skill-pin load REL [ROOT] | skill-pin pin REL [ROOT] | skill-pin verify [ROOT] | skill-pin verify-one REL [ROOT] | skill-pin tip [ROOT] | skill-pin log [ROOT] | skill-pin height [ROOT] | skill-pin get HEIGHT [ROOT] | skill-pin at HDR [ROOT] | skill-pin range FROM TO [ROOT] | skill-pin since HDR [ROOT] | skill-pin until HDR [ROOT] | skill-pin before HDR [ROOT] | skill-pin after HDR [ROOT] | skill-pin first [ROOT]",
        file=sys.stderr,
    )
    return 2
