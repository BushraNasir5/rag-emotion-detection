from __future__ import annotations

import hashlib
from typing import Iterable, List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


class Embedder:
    """Wrapper with a deterministic fallback for offline runs."""

    def __init__(self, model_name: str, normalize: bool = True) -> None:
        self.normalize = normalize
        self.model_name = model_name
        self._model = None
        if SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer(model_name)
            except Exception:
                self._model = None

    def _fallback(self, texts: Iterable[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for text in texts:
            bucket = np.zeros(128, dtype=np.float32)
            for token in text.lower().split():
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                bucket[int(digest, 16) % 128] += 1.0
            vectors.append(bucket)
        out = np.stack(vectors, axis=0)
        if self.normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
            out = out / norms
        return out

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 128), dtype=np.float32)
        if self._model is None:
            return self._fallback(texts)
        out = self._model.encode(texts, normalize_embeddings=self.normalize)
        return np.asarray(out, dtype=np.float32)
