from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earag.data import load_jsonl
from earag.embeddings import Embedder
from earag.retriever import FaissRetriever


def normalize_emotion(context: str) -> str:
    c = context.lower()
    if c in {"angry", "afraid", "sad", "anxious", "disappointed", "furious"}:
        return "distress"
    if c in {"excited", "proud", "grateful", "joyful", "content"}:
        return "positive"
    return "neutral"


def tone_for_emotion(emotion: str) -> str:
    if emotion == "distress":
        return "supportive"
    if emotion == "positive":
        return "celebratory"
    return "neutral"


def map_emotion(label: str) -> str:
    x = label.lower()
    if x in {"anger", "fear", "sadness", "disgust", "nervousness", "grief", "distress"}:
        return "distress"
    if x in {"joy", "gratitude", "optimism", "love", "excitement", "admiration", "positive"}:
        return "positive"
    return "neutral"


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def select_gold_context(query: str, emotion: str, kb_rows: list[dict], top_k: int = 3) -> list[str]:
    qtok = tokens(query)
    scored = []
    for row in kb_rows:
        dtok = tokens(row["text"])
        overlap = len(qtok & dtok) / max(len(qtok), 1)
        emo_bonus = 0.20 if row.get("emotion_hint", "neutral") == emotion else 0.0
        score = overlap + emo_bonus
        scored.append((score, row["id"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc_id for _, doc_id in scored[:top_k]]


def main() -> None:
    kb_path = ROOT / "data" / "processed" / "knowledge_base.jsonl"
    if not kb_path.exists():
        raise FileNotFoundError("Knowledge base missing. Run scripts/prepare_real_datasets.py first.")
    kb_rows = load_jsonl(kb_path)
    embedder = Embedder(model_name="BAAI/bge-large-en-v1.5", normalize=True)
    retriever = FaissRetriever(embedder)
    retriever.build(kb_rows)

    def auto_gold(query: str, emotion: str) -> list[str]:
        docs = retriever.search(query, top_k=3)
        ids = [d.doc_id for d in docs]
        if ids:
            return ids
        return select_gold_context(query, emotion, kb_rows)

    ds = None
    if load_dataset is not None:
        try:
            ds = load_dataset("empathetic_dialogues")["validation"]
        except Exception:
            ds = None

    go_fallback = None
    if load_dataset is not None and ds is None:
        try:
            go_fallback = load_dataset("go_emotions", "raw")["test"]
        except Exception:
            go_fallback = None

    output = ROOT / "data" / "custom_eval" / "ea_rag_eval.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if ds is not None:
        for i, item in enumerate(ds):
            if i >= 200:
                break
            query = item["prompt"]
            emotion = normalize_emotion(item.get("context", "neutral"))
            gold_context = auto_gold(query, emotion)
            rows.append(
                {
                    "id": f"real_q_{i}",
                    "query": query,
                    "target_emotion": emotion,
                    "gold_context": gold_context,
                    "reference_answer": item["utterance"],
                    "tone_constraint": tone_for_emotion(emotion),
                }
            )
    elif go_fallback is not None:
        emotion_columns = [
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
            "neutral",
        ]
        for i, item in enumerate(go_fallback):
            if i >= 200:
                break
            best_emo = "neutral"
            best_score = -1
            for emo in emotion_columns:
                s = int(item.get(emo, 0))
                if s > best_score:
                    best_score = s
                    best_emo = emo
            emotion = map_emotion(best_emo)
            query = item["text"]
            rows.append(
                {
                    "id": f"go_q_{i}",
                    "query": query,
                    "target_emotion": emotion,
                    "gold_context": auto_gold(query, emotion),
                    "reference_answer": "Reference response placeholder.",
                    "tone_constraint": tone_for_emotion(emotion),
                }
            )
    else:
        fallback = [
            ("I am stressed and cannot sleep before exams.", "distress"),
            ("I finished my project milestone today!", "positive"),
            ("Can you explain RAG evaluation metrics?", "neutral"),
        ]
        for i, (q, emo) in enumerate(fallback):
            rows.append(
                {
                    "id": f"fallback_q_{i}",
                    "query": q,
                    "target_emotion": emo,
                    "gold_context": auto_gold(q, emo),
                    "reference_answer": "Reference response placeholder.",
                    "tone_constraint": tone_for_emotion(emo),
                }
            )
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Wrote {len(rows)} examples to {output}")


if __name__ == "__main__":
    main()
