# Emotion-Aware Retrieval-Augmented Generation (EA-RAG)

This repository implements an end-to-end EA-RAG framework for research and final-year project use.

## What is included

- Reproducible baseline RAG pipeline with FAISS retrieval.
- Emotion-aware modules:
  - emotion classification
  - emotion-conditioned query rewriting
  - emotion-aware ranking
  - tone-conditioned generation prompt
- Experiment runners for:
  - baseline
  - full EA-RAG
  - ablations
- Evaluation scripts for retrieval, generation, and emotion alignment.
- Paper-writing assets (method, results templates, and architecture figure).

## Project structure

- `configs/`: Dataset, model, and experiment configuration.
- `src/earag/`: Core source code.
- `scripts/`: Command-line entry points.
- `data/`: Dataset placeholders and small demo benchmark.
- `results/`: Experiment outputs (metrics and logs).
- `paper/`: Draft manuscript sections and tables.

## Quick start

1. Create env and install dependencies:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Prepare data placeholders and sample benchmark:
   - `python scripts/prepare_datasets.py`
3. Run baseline:
   - `python scripts/run_baseline.py`
4. Run full EA-RAG:
   - `python scripts/run_earag.py`
5. Run ablations:
   - `python scripts/run_ablation.py`
6. Prepare real datasets and camera-ready summary:
   - `python scripts/prepare_real_datasets.py`
   - `python scripts/generate_custom_eval_from_empathetic.py`
   - `python scripts/run_multiseed.py`
   - `python scripts/generate_camera_ready_tables.py`

## Notes

- The project is config-driven via YAML files in `configs/`.
- External datasets (GoEmotions, SemEval, EmpatheticDialogues) are referenced with download instructions in `data/README.md`.
- LLM calls are adapter-based. By default, a deterministic local mock generator is used for reproducible smoke testing.
