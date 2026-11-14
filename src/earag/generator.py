from __future__ import annotations

from typing import List

from .retriever import RetrievedDoc


class LocalMockGenerator:
    """Deterministic fallback generator for reproducibility."""

    def generate(self, query: str, docs: List[RetrievedDoc], tone: str = "neutral") -> str:
        context = " ".join(d.text for d in docs[:2])
        if tone == "supportive":
            prefix = "I understand this may feel difficult. "
        elif tone == "calm":
            prefix = "Let us go step by step. "
        elif tone == "celebratory":
            prefix = "That is great to hear. "
        else:
            prefix = ""
        return f"{prefix}Based on retrieved knowledge: {context} | Response to query: {query}"
