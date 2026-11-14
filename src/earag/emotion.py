from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

try:
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None


@dataclass
class EmotionPrediction:
    label: str
    score: float


class EmotionDetector:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._clf = None
        if pipeline is not None:
            try:
                self._clf = pipeline("text-classification", model=model_name, top_k=1)
            except Exception:
                self._clf = None

    def _fallback(self, text: str) -> EmotionPrediction:
        t = text.lower()
        if any(k in t for k in ["overwhelmed", "frustrated", "sad", "anxious", "stress"]):
            return EmotionPrediction(label="distress", score=0.8)
        if any(k in t for k in ["great", "happy", "excited", "completed"]):
            return EmotionPrediction(label="positive", score=0.8)
        return EmotionPrediction(label="neutral", score=0.7)

    def predict(self, text: str) -> EmotionPrediction:
        if self._clf is None:
            return self._fallback(text)
        result = self._clf(text)[0][0]
        label = result["label"].lower()
        if label in {"anger", "fear", "sadness", "disgust"}:
            mapped = "distress"
        elif label in {"joy"}:
            mapped = "positive"
        else:
            mapped = "neutral"
        return EmotionPrediction(label=mapped, score=float(result["score"]))
