# Experimental Setup

## Datasets

- Primary benchmark targets: GoEmotions, SemEval-2018 Task 1, EmpatheticDialogues.
- Reproducible demo benchmark used in this run: `data/custom_eval/ea_rag_eval.jsonl`.
- Knowledge base for smoke experiments: `data/processed/knowledge_base.jsonl`.

## Models

- Emotion classifier: `j-hartmann/emotion-english-distilroberta-base` (with deterministic fallback).
- Retriever embedding: `BAAI/bge-large-en-v1.5` (with deterministic fallback encoder).
- Generator: local deterministic adapter for reproducible testing.

## Metrics

- Retrieval: `hit@1`, `hit@3`, `hit@5`.
- RAG quality proxies: faithfulness proxy, answer relevancy proxy, context precision proxy.
- Emotion quality: classifier-based emotion alignment accuracy.

## Results Snapshot (current run)

- Baseline retrieval: `hit@1=0.5`, `hit@3=1.0`, `hit@5=1.0`.
- EA-RAG retrieval: `hit@1=0.5`, `hit@3=1.0`, `hit@5=1.0`.
- Emotion alignment: `0.5` on 2 sample prompts.
- Ablation indicates ranking module contributes most in this demo (`no_emotion_ranking` drops `hit@1` to `0.0`).

## Limitations of current run

- Demo benchmark is intentionally small for smoke validation.
- Full claims should be made only after running on complete GoEmotions/SemEval/EmpatheticDialogues splits and human evaluation protocol.

## Camera-Ready Protocol

1. Build real knowledge base:
   - `python scripts/prepare_real_datasets.py`
2. Build larger custom evaluation set:
   - `python scripts/generate_custom_eval_from_empathetic.py`
3. Run repeated-seed experiments:
   - `python scripts/run_multiseed.py`
4. Export paper tables:
   - `python scripts/generate_camera_ready_tables.py`

Outputs:
- `results/multiseed_summary.json`
- `paper/tables/results_main_camera_ready.csv`
- `paper/tables/results_main_camera_ready.md`
