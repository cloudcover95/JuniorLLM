"""Dilithium-shaped API. secure=False. Do not use on a chain."""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass


@dataclass(frozen=True)
class ToyKey:
    sk: bytes
    pk: bytes
    secure: bool = False
    scheme: str = "junior-toy-not-mldsa"


def keygen(seed: bytes = b"junior-toy") -> ToyKey:
    sk = hashlib.sha256(seed).digest()
    pk = hashlib.sha256(sk + b"pk").digest()
    return ToyKey(sk, pk)


def sign(key: ToyKey, msg: bytes) -> bytes:
    return hmac.new(key.sk, msg, hashlib.sha256).digest()


def verify(key: ToyKey, msg: bytes, sig: bytes) -> bool:
    return hmac.compare_digest(sign(key, msg), sig)
