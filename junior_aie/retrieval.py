"""Retrieval stack — chunker + BM25 + dense + reranker."""
from __future__ import annotations

import math
from collections import Counter

from junior_aie.embed import cosine, embed, tokenize


def chunk(text: str, size: int = 40, overlap: int = 8) -> list[str]:
    words = (text or "").split()
    if not words:
        return []
    out = []
    i = 0
    step = max(1, size - overlap)
    while i < len(words):
        out.append(" ".join(words[i : i + size]))
        i += step
    return out


class RetrievalStack:
    def __init__(self) -> None:
        self.docs: list[str] = []
        self.vecs: list[list[float]] = []
        self.df: Counter[str] = Counter()

    def add(self, text: str) -> None:
        for c in chunk(text):
            self.docs.append(c)
            self.vecs.append(embed(c))
            for t in set(tokenize(c)):
                self.df[t] += 1

    def _bm25(self, query: str, doc: str, k1: float = 1.5, b: float = 0.75) -> float:
        q = tokenize(query)
        d = tokenize(doc)
        if not d:
            return 0.0
        tf = Counter(d)
        avgdl = sum(len(tokenize(x)) for x in self.docs) / max(1, len(self.docs))
        n = len(self.docs)
        score = 0.0
        for term in q:
            f = tf.get(term, 0)
            if not f:
                continue
            n_q = self.df.get(term, 0)
            idf = math.log((n - n_q + 0.5) / (n_q + 0.5) + 1.0)
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * len(d) / max(1.0, avgdl)))
        return score

    def search(self, query: str, k: int = 4) -> list[tuple[str, float]]:
        if not self.docs:
            return []
        qv = embed(query)
        scored = []
        for doc, vec in zip(self.docs, self.vecs):
            dense = cosine(qv, vec)
            sparse = self._bm25(query, doc)
            scored.append((doc, 0.6 * dense + 0.4 * sparse))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
