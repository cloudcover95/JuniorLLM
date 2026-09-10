"""Local JUNIT node. Loopback-first."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from bitnet_net.tx import Tx


@dataclass
class Block:
    height: int
    prev: str
    txs: list[dict]
    hdr: str
    pq_hdr: str | None = None


class Node:
    def __init__(self, root: Path, name: str = "local"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.balances: dict[str, int] = {}
        self.nonces: dict[str, int] = {}
        self.blocks: list[Block] = []
        self.mempool: list[Tx] = []
        self._pq = None
        try:
            from bitnet_pq.chain import Chain

            self._pq = Chain()
            self._pq.genesis("net", [1, 0, -1, 1] * 8)
        except Exception:
            self._pq = None
        self._load()

    def _path(self) -> Path:
        return self.root / "chain.jsonl"

    def _load(self) -> None:
        if not self._path().exists():
            return
        for line in self._path().read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            self.blocks.append(Block(**d))
            for t in d["txs"]:
                self._apply(Tx(t["frm"], t["to"], t["amount"], t["nonce"]), check=False)

    def _apply(self, tx: Tx, check: bool = True) -> None:
        if tx.amount <= 0:
            raise ValueError("amount")
        if tx.frm == "mint":
            self.balances[tx.to] = self.balances.get(tx.to, 0) + tx.amount
            return
        if check and self.balances.get(tx.frm, 0) < tx.amount:
            raise ValueError("insufficient")
        if check and self.nonces.get(tx.frm, 0) != tx.nonce:
            raise ValueError("nonce")
        self.balances[tx.frm] = self.balances.get(tx.frm, 0) - tx.amount
        self.balances[tx.to] = self.balances.get(tx.to, 0) + tx.amount
        self.nonces[tx.frm] = tx.nonce + 1

    def mint(self, to: str, amount: int) -> Tx:
        tx = Tx("mint", to, amount, 0)
        self.mempool.append(tx)
        return tx

    def transfer(self, frm: str, to: str, amount: int) -> Tx:
        tx = Tx(frm, to, amount, self.nonces.get(frm, 0))
        self._apply(tx)  # check; revert via copy would be nicer but we apply then commit on seal
        # undo until seal — keep simple: apply only at seal. Re-check:
        self.balances[frm] += amount
        self.balances[to] -= amount
        self.nonces[frm] -= 1
        if self.nonces[frm] < 0:
            self.nonces[frm] = 0
        self.mempool.append(tx)
        return tx

    def seal(self) -> Block:
        prev = self.blocks[-1].hdr if self.blocks else "0" * 64
        txs = [t.as_dict() for t in self.mempool]
        for t in self.mempool:
            self._apply(t)
        self.mempool.clear()
        pq_hdr = None
        if self._pq is not None:
            try:
                pq_hdr = self._pq.tick("net").hdr
            except Exception:
                pq_hdr = None
        body = prev + json.dumps(txs, sort_keys=True) + (pq_hdr or "")
        hdr = hashlib.sha256(body.encode()).hexdigest()
        blk = Block(len(self.blocks), prev, txs, hdr, pq_hdr)
        self.blocks.append(blk)
        with self._path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(blk.__dict__) + "\n")
        return blk

    def gossip(self) -> dict:
        return {
            "format": "junior-gossip-v1",
            "kind": "bitnet-net-blocks",
            "node": self.name,
            "count": len(self.blocks),
            "tip": self.blocks[-1].hdr if self.blocks else None,
            "blocks": [b.__dict__ for b in self.blocks],
        }
