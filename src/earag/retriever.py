from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

from .embeddings import Embedder


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    emotion_hint: str


class FaissRetriever:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.docs: List[Dict] = []
        self.index = None
        self.matrix: np.ndarray | None = None

    def build(self, docs: List[Dict]) -> None:
        self.docs = docs
        texts = [d["text"] for d in docs]
        self.matrix = self.embedder.encode(texts)
        if faiss is not None:
            dim = self.matrix.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.matrix)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedDoc]:
        if self.matrix is None:
            raise RuntimeError("Retriever index not built.")
        q = self.embedder.encode([query])
        if self.index is not None:
            scores, idx = self.index.search(q, top_k)
            scored = zip(idx[0].tolist(), scores[0].tolist())
        else:
            sims = (self.matrix @ q[0]).tolist()
            order = np.argsort(-np.asarray(sims))[:top_k].tolist()
            scored = [(i, sims[i]) for i in order]

        out: List[RetrievedDoc] = []
        for i, s in scored:
            d = self.docs[i]
            out.append(
                RetrievedDoc(
                    doc_id=d["id"],
                    text=d["text"],
                    score=float(s),
                    emotion_hint=d.get("emotion_hint", "neutral"),
                )
            )
        return out
