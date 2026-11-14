from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earag.data import load_jsonl, save_json
from earag.emotion import EmotionDetector


def main() -> None:
    target_rows = {r["id"]: r for r in load_jsonl(ROOT / "data" / "custom_eval" / "ea_rag_eval.jsonl")}
    preds = json.loads((ROOT / "results" / "earag" / "predictions.json").read_text(encoding="utf-8"))
    detector = EmotionDetector("j-hartmann/emotion-english-distilroberta-base")
    correct = 0
    for p in preds:
        target = target_rows[p["id"]]["target_emotion"]
        pred = detector.predict(p["answer"]).label
        if pred == target:
            correct += 1
    score = correct / len(preds) if preds else 0.0
    payload = {"emotion_alignment_accuracy": score, "n": len(preds)}
    save_json(ROOT / "results" / "earag" / "emotion_metrics.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
