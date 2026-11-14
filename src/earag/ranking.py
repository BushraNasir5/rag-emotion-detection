from __future__ import annotations

from typing import List

from .retriever import RetrievedDoc


def emotion_alignment(doc_emotion: str, query_emotion: str) -> float:
    return 1.0 if doc_emotion == query_emotion else 0.2


def safety_priority(query_emotion: str) -> float:
    return 1.0 if query_emotion == "distress" else 0.3


def rerank_docs(
    docs: List[RetrievedDoc],
    query_emotion: str,
    alpha: float,
    beta: float,
    gamma: float,
) -> List[RetrievedDoc]:
    scored = []
    for d in docs:
        final_score = (
            alpha * d.score
            + beta * emotion_alignment(d.emotion_hint, query_emotion)
            + gamma * safety_priority(query_emotion)
        )
        scored.append((final_score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored]
