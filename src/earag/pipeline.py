from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .emotion import EmotionDetector
from .generator import LocalMockGenerator
from .query_rewriter import rewrite_query
from .ranking import rerank_docs
from .retriever import FaissRetriever


@dataclass
class PipelineOutput:
    query_id: str
    retrieved_ids: List[str]
    answer: str


class BaselineRAGPipeline:
    def __init__(self, retriever: FaissRetriever, generator: LocalMockGenerator, top_k: int = 5) -> None:
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def run_one(self, item: Dict) -> PipelineOutput:
        docs = self.retriever.search(item["query"], top_k=self.top_k)
        answer = self.generator.generate(item["query"], docs, tone="neutral")
        return PipelineOutput(
            query_id=item["id"],
            retrieved_ids=[d.doc_id for d in docs],
            answer=answer,
        )


class EARAGPipeline:
    def __init__(
        self,
        retriever: FaissRetriever,
        generator: LocalMockGenerator,
        emotion_detector: EmotionDetector,
        top_k: int = 5,
        alpha: float = 0.70,
        beta: float = 0.25,
        gamma: float = 0.05,
        use_query_rewrite: bool = True,
        use_emotion_ranking: bool = True,
        use_tone_conditioning: bool = True,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.emotion_detector = emotion_detector
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_query_rewrite = use_query_rewrite
        self.use_emotion_ranking = use_emotion_ranking
        self.use_tone_conditioning = use_tone_conditioning

    def _tone_from_emotion(self, emotion: str) -> str:
        if emotion == "distress":
            return "supportive"
        if emotion == "positive":
            return "celebratory"
        return "neutral"

    def run_one(self, item: Dict) -> PipelineOutput:
        emo = self.emotion_detector.predict(item["query"]).label
        query = rewrite_query(item["query"], emo) if self.use_query_rewrite else item["query"]
        docs = self.retriever.search(query, top_k=self.top_k)
        if self.use_emotion_ranking:
            docs = rerank_docs(docs, emo, self.alpha, self.beta, self.gamma)
        tone = self._tone_from_emotion(emo) if self.use_tone_conditioning else "neutral"
        answer = self.generator.generate(item["query"], docs, tone=tone)
        return PipelineOutput(
            query_id=item["id"],
            retrieved_ids=[d.doc_id for d in docs],
            answer=answer,
        )
