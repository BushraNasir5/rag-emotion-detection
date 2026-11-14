from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earag.config import load_yaml
from earag.data import load_jsonl, save_json
from earag.embeddings import Embedder
from earag.emotion import EmotionDetector
from earag.generator import LocalMockGenerator
from earag.metrics import simple_retrieval_metrics
from earag.pipeline import EARAGPipeline
from earag.retriever import FaissRetriever


def main() -> None:
    exp_cfg = load_yaml(ROOT / "configs" / "experiment.yaml")["experiment"]
    model_cfg = load_yaml(ROOT / "configs" / "models.yaml")["models"]

    docs = load_jsonl(ROOT / "data" / "processed" / "knowledge_base.jsonl")
    eval_rows = load_jsonl(ROOT / "data" / "custom_eval" / "ea_rag_eval.jsonl")

    embedder = Embedder(
        model_name=model_cfg["embedding"]["primary"],
        normalize=model_cfg["embedding"]["normalize"],
    )
    retriever = FaissRetriever(embedder)
    retriever.build(docs)
    generator = LocalMockGenerator()
    detector = EmotionDetector(model_cfg["emotion_classifier"]["name"])
    w = exp_cfg["retrieval_weights"]
    ab = exp_cfg["ablations"]

    pipeline = EARAGPipeline(
        retriever=retriever,
        generator=generator,
        emotion_detector=detector,
        top_k=exp_cfg["top_k"],
        alpha=w["alpha"],
        beta=w["beta"],
        gamma=w["gamma"],
        use_query_rewrite=not ab["remove_query_rewrite"],
        use_emotion_ranking=not ab["remove_emotion_ranking"],
        use_tone_conditioning=not ab["remove_tone_conditioning"],
    )

    predictions = []
    metric_rows = []
    for item in eval_rows:
        out = pipeline.run_one(item)
        predictions.append(
            {
                "id": out.query_id,
                "answer": out.answer,
                "retrieved_ids": out.retrieved_ids,
            }
        )
        metric_rows.append({"retrieved_ids": out.retrieved_ids, "gold_ids": item["gold_context"]})
    metrics = simple_retrieval_metrics(metric_rows)

    out_dir = ROOT / "results" / "earag"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "metrics.json", metrics)
    with (out_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
