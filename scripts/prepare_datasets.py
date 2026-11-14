from __future__ import annotations

import json
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    docs = [
        {
            "id": "d1",
            "text": "Breathing exercises can reduce acute stress and help regulate anxiety.",
            "emotion_hint": "distress",
        },
        {
            "id": "d2",
            "text": "Structured study plans improve exam performance through consistency.",
            "emotion_hint": "neutral",
        },
        {
            "id": "d3",
            "text": "Celebrating small wins reinforces motivation and long-term habits.",
            "emotion_hint": "positive",
        },
        {
            "id": "d4",
            "text": "When users are frustrated, acknowledge feelings before proposing steps.",
            "emotion_hint": "distress",
        },
    ]
    eval_rows = [
        {
            "id": "q1",
            "query": "I am overwhelmed before my exam and cannot focus.",
            "target_emotion": "distress",
            "gold_context": ["d1", "d2"],
            "reference_answer": "Acknowledge stress and suggest breathing and planning.",
            "tone_constraint": "supportive",
        },
        {
            "id": "q2",
            "query": "I completed my milestones this week.",
            "target_emotion": "positive",
            "gold_context": ["d3"],
            "reference_answer": "Congratulate and encourage consistency.",
            "tone_constraint": "celebratory",
        },
    ]
    write_jsonl(Path("data/processed/knowledge_base.jsonl"), docs)
    write_jsonl(Path("data/custom_eval/ea_rag_eval.jsonl"), eval_rows)
    print("Prepared demo knowledge base and custom eval dataset.")


if __name__ == "__main__":
    main()
