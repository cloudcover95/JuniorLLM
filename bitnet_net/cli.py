#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

from bitnet_net.node import Node

ROOT = Path.home() / ".juniorllm" / "bitnet_net"


def node() -> Node:
    return Node(ROOT)


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "help"
    n = node()
    if cmd == "mint":
        n.mint(argv[2], int(argv[3]))
        print(n.seal().__dict__)
        return 0
    if cmd == "send":
        n.transfer(argv[2], argv[3], int(argv[4]))
        print(n.seal().__dict__)
        return 0
    if cmd == "bal":
        print(json.dumps(n.balances, indent=2))
        return 0
    if cmd == "gossip":
        print(json.dumps(n.gossip())[:800])
        return 0
    print("mint <to> <amt> | send <from> <to> <amt> | bal | gossip")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
