from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earag.config import load_yaml

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def map_emotion(label: str) -> str:
    label = label.lower()
    if label in {"anger", "fear", "sadness", "disgust", "nervousness", "grief"}:
        return "distress"
    if label in {"joy", "gratitude", "optimism", "love", "excitement", "admiration"}:
        return "positive"
    return "neutral"


def build_goemotions(path: Path, limit: int = 2000) -> int:
    if load_dataset is None:
        return 0
    ds = load_dataset("go_emotions", "raw")
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
    rows = []
    for i, item in enumerate(ds["train"]):
        if i >= limit:
            break
        label = "neutral"
        best_score = -1
        for emo in emotion_columns:
            score = int(item.get(emo, 0))
            if score > best_score:
                best_score = score
                label = emo
        rows.append(
            {
                "id": f"go_{i}",
                "text": item["text"],
                "emotion_hint": map_emotion(label),
                "source": "goemotions",
            }
        )
    write_jsonl(path, rows)
    return len(rows)


def build_empathetic(path: Path, limit: int = 2000) -> int:
    if load_dataset is None:
        return 0
    try:
        ds = load_dataset("empathetic_dialogues")
    except Exception:
        return 0
    rows = []
    for i, item in enumerate(ds["train"]):
        if i >= limit:
            break
        rows.append(
            {
                "id": f"ed_{i}",
                "text": item["utterance"],
                "emotion_hint": map_emotion(item.get("context", "neutral")),
                "source": "empathetic_dialogues",
            }
        )
    write_jsonl(path, rows)
    return len(rows)


def load_semeval_local(path: Path, limit: int = 2000) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                text = row.get("text", "") or row.get("tweet", "") or row.get("Tweet", "")
                emotion = row.get("emotion", "neutral") or row.get("label", "neutral")
                if not text:
                    continue
                rows.append(
                    {
                        "id": f"sem_{i}",
                        "text": text,
                        "emotion_hint": map_emotion(emotion),
                        "source": "semeval2018",
                    }
                )
    except Exception:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                if len(row) < 2:
                    continue
                text = row[1].strip()
                emotion = row[2].strip() if len(row) > 2 else "neutral"
                if i == 0 and text.lower() in {"text", "tweet"}:
                    continue
                if not text:
                    continue
                rows.append(
                    {
                        "id": f"sem_{i}",
                        "text": text,
                        "emotion_hint": map_emotion(emotion),
                        "source": "semeval2018",
                    }
                )
    return rows


def combine_knowledge_base() -> None:
    cfg = load_yaml(ROOT / "configs" / "datasets.yaml")["datasets"]
    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    go_path = processed / "goemotions_kb.jsonl"
    ed_path = processed / "empathetic_kb.jsonl"
    semeval_train = ROOT / cfg["semeval2018"]["train_path"]

    go_n = build_goemotions(go_path)
    ed_n = build_empathetic(ed_path)
    sem_rows = load_semeval_local(semeval_train)

    combined = []
    for p in [go_path, ed_path]:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        combined.append(json.loads(line))
    combined.extend(sem_rows)
    if not combined:
        raise RuntimeError("No real datasets available. Check network or local SemEval CSV files.")

    write_jsonl(processed / "knowledge_base.jsonl", combined)
    meta = {
        "goemotions_records": go_n,
        "empathetic_dialogues_records": ed_n,
        "semeval_records": len(sem_rows),
        "total_knowledge_records": len(combined),
    }
    (processed / "dataset_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    combine_knowledge_base()
